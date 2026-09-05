import os
import json
import re
from collections import Counter, defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# --- 1. Constants & Semantic Class Definitions ---
# 6 land-cover classes in SECOND + Class 0 for No Change / Background
CLASS_NAMES = [
    "no_change",                    # Class 0: (255, 255, 255)
    "non-vegetated ground surface", # Class 1: (128, 128, 128)
    "tree",                         # Class 2: (0, 255, 0)
    "low vegetation",               # Class 3: (0, 128, 0)
    "water",                        # Class 4: (0, 0, 255)
    "buildings",                    # Class 5: (128, 0, 0)
    "playgrounds"                   # Class 6: (255, 0, 0)
]

# RGB color to class index mapping for SECOND dataset
RGB_TO_CLASS = {
    (255, 255, 255): 0, # No change
    (128, 128, 128): 1, # NVG surface
    (0, 255, 0):     2, # Tree
    (0, 128, 0):     3, # Low vegetation
    (0, 0, 255):     4, # Water
    (128, 0, 0):     5, # Buildings
    (255, 0, 0):     6, # Playgrounds
}

# The 19 closed-vocabulary answers across CDVQA
ANSWER_VOCAB = [
    'no', 'yes',
    'NVG_surface', 'trees', 'low_vegetation', 'water', 'buildings', 'playgrounds',
    '0', '0_to_10', '10_to_20', '20_to_30', '30_to_40', '40_to_50',
    '50_to_60', '60_to_70', '70_to_80', '80_to_90', '90_to_100'
]
ANS_TO_IDX = {ans: i for i, ans in enumerate(ANSWER_VOCAB)}
IDX_TO_ANS = {i: ans for i, ans in enumerate(ANSWER_VOCAB)}


# --- 2. Color Mask Conversion ---
def rgb_mask_to_class_indices(rgb_mask: np.ndarray) -> np.ndarray:
    """
    Converts (H, W, 3) RGB label image into (H, W) uint8 class index map (0 to 6).
    """
    h, w, _ = rgb_mask.shape
    label_mask = np.zeros((h, w), dtype=np.int64)
    for rgb, cls_idx in RGB_TO_CLASS.items():
        match = (rgb_mask[:, :, 0] == rgb[0]) & \
                (rgb_mask[:, :, 1] == rgb[1]) & \
                (rgb_mask[:, :, 2] == rgb[2])
        label_mask[match] = cls_idx
    return label_mask


# --- 3. Text Tokenizer for CDVQA Questions ---
class QuestionTokenizer:
    """
    Lightweight rule-based tokenizer for CDVQA questions.
    """
    def __init__(self, vocab_list=None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.unk_idx = 1
        
        if vocab_list is None:
            words = [
                'area', 'areas', 'buildings', 'change', 'changed', 'decrease', 
                'decreased', 'did', 'event', 'first', 'ground', 'has', 'have', 
                'how', 'image', 'imagery', 'in', 'increase', 'increased', 'is', 
                'largest', 'low', 'mainly', 'much', 'non', 'not', 'of', 'percentage', 
                'playgrounds', 'post', 'pre', 'proportion', 'ratio', 'regions', 
                'second', 'smallest', 'surface', 'the', 'to', 'trees', 'type', 
                'unchanged', 'vegetated', 'vegetation', 'water', 'what'
            ]
        else:
            words = vocab_list
            
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def tokenize(self, text: str) -> list:
        clean = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower()).split()
        return clean

    def encode(self, text: str, max_len: int = 24) -> torch.Tensor:
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens][:max_len]
        padding = [self.pad_idx] * (max_len - len(ids))
        return torch.tensor(ids + padding, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


# --- 4. Subsampling & Stratification Utility ---
def create_stratified_subset(
    cdvqa_dir: str,
    target_count: int = 800,
    save_path: str = None
) -> list:
    """
    Subsamples target_count (400-800) image pairs from CDVQA train images,
    stratifying across the 6 land-cover change classes.
    """
    train_img_json = os.path.join(cdvqa_dir, "Train_images.json")
    train_q_json = os.path.join(cdvqa_dir, "Train_questions.json")
    train_a_json = os.path.join(cdvqa_dir, "Train_answers.json")

    with open(train_img_json, 'r') as f:
        images_data = json.load(f)['images']
    with open(train_q_json, 'r') as f:
        questions_data = json.load(f)['questions']
    with open(train_a_json, 'r') as f:
        answers_data = json.load(f)['answers']

    q_map = {q['id']: q for q in questions_data}
    a_map = {a['question_id']: a for a in answers_data}

    # Group questions by unique file_name
    file_to_qa = defaultdict(list)
    for entry in images_data:
        fname = entry['file_name']
        for qid in entry['questions_ids']:
            if qid in q_map and qid in a_map:
                file_to_qa[fname].append({
                    'question_id': qid,
                    'question': q_map[qid]['question'],
                    'type': q_map[qid]['type'],
                    'answer': a_map[qid]['answer']
                })

    # Deduplicate questions per filename
    for fname in file_to_qa:
        seen = set()
        deduped = []
        for item in file_to_qa[fname]:
            if item['question_id'] not in seen:
                seen.add(item['question_id'])
                deduped.append(item)
        file_to_qa[fname] = deduped

    class_keywords = {
        1: ["non-vegetated", "ground", "nvg_surface"],
        2: ["tree", "trees"],
        3: ["low vegetation", "low_vegetation"],
        4: ["water"],
        5: ["building", "buildings"],
        6: ["playground", "playgrounds"]
    }

    file_classes = defaultdict(set)
    for fname, qa_list in file_to_qa.items():
        for qa in qa_list:
            ans = qa['answer'].lower()
            q_text = qa['question'].lower()
            for cls_id, kws in class_keywords.items():
                if any(kw in ans or (kw in q_text and ans == 'yes') for kw in kws):
                    file_classes[fname].add(cls_id)

    selected_files = set()
    per_class_target = target_count // 6

    # Select samples for each class
    for cls_id in range(1, 7):
        candidates = [f for f, clss in file_classes.items() if cls_id in clss and f not in selected_files]
        candidates.sort()
        np.random.seed(42 + cls_id)
        chosen = np.random.choice(candidates, size=min(per_class_target, len(candidates)), replace=False)
        selected_files.update(chosen)

    # Fill up to target_count (if target_count <= 0, take all files)
    all_files = sorted(list(file_to_qa.keys()))
    for f in all_files:
        if 0 < target_count <= len(selected_files):
            break
        selected_files.add(f)

    subset = [{'file_name': f, 'qa_pairs': file_to_qa[f]} for f in sorted(list(selected_files))]

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(subset, f, indent=2)
        print(f"Saved stratified subset of {len(subset)} files to {save_path}")

    return subset


# --- 5. PyTorch Dataset Class with In-Memory Caching ---
class CDVQADataset(Dataset):
    """
    Bi-temporal Change Detection + Change-VQA Dataset with RAM Caching.
    Eliminates disk I/O bottlenecks to keep RTX 4050 compute pinned at 100%.
    """
    def __init__(
        self,
        im1_dir: str,
        im2_dir: str,
        label1_dir: str,
        label2_dir: str,
        cdvqa_dir: str,
        split: str = 'Train',
        image_size: int = 256,
        subsample_count: int = 800,
        tokenizer: QuestionTokenizer = None,
        augment: bool = True,
        cache_in_ram: bool = True
    ):
        self.im1_dir = im1_dir
        self.im2_dir = im2_dir
        self.label1_dir = label1_dir
        self.label2_dir = label2_dir
        self.image_size = image_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.tokenizer = tokenizer or QuestionTokenizer()

        subset_cache = os.path.join(cdvqa_dir, f"{split.lower()}_subset_{subsample_count}.json")
        if split.lower() == 'train':
            if os.path.exists(subset_cache):
                with open(subset_cache, 'r') as f:
                    file_records = json.load(f)
            else:
                file_records = create_stratified_subset(cdvqa_dir, subsample_count, subset_cache)
        else:
            file_records = self._load_split_records(cdvqa_dir, split, max_files=subsample_count)

        self.samples = []
        unique_fnames = set()
        for rec in file_records:
            fname = rec['file_name']
            if not os.path.exists(os.path.join(self.im1_dir, fname)):
                continue
            unique_fnames.add(fname)
            for qa in rec['qa_pairs']:
                if qa['answer'] in ANS_TO_IDX:
                    self.samples.append({
                        'file_name': fname,
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'ans_idx': ANS_TO_IDX[qa['answer']]
                    })

        print(f"CDVQADataset [{split}]: {len(unique_fnames)} unique image pairs, {len(self.samples)} QA samples.")

        # In-Memory RAM Caching
        self.image_cache = {}
        if self.cache_in_ram:
            print(f"--> Pre-caching {len(unique_fnames)} image pairs into RAM for zero-latency GPU throughput...")
            for fname in unique_fnames:
                im1 = cv2.imread(os.path.join(self.im1_dir, fname))
                im2 = cv2.imread(os.path.join(self.im2_dir, fname))
                l1 = cv2.imread(os.path.join(self.label1_dir, fname))
                l2 = cv2.imread(os.path.join(self.label2_dir, fname))

                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
                l1 = cv2.cvtColor(l1, cv2.COLOR_BGR2RGB)
                l2 = cv2.cvtColor(l2, cv2.COLOR_BGR2RGB)

                im1 = cv2.resize(im1, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                im2 = cv2.resize(im2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                l1 = cv2.resize(l1, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                l2 = cv2.resize(l2, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

                m1 = rgb_mask_to_class_indices(l1)
                m2 = rgb_mask_to_class_indices(l2)

                self.image_cache[fname] = (im1, im2, m1, m2)
            print("--> Caching completed successfully.")

    def _load_split_records(self, cdvqa_dir: str, split: str, max_files: int = 400) -> list:
        img_p = os.path.join(cdvqa_dir, f"{split}_images.json")
        q_p = os.path.join(cdvqa_dir, f"{split}_questions.json")
        a_p = os.path.join(cdvqa_dir, f"{split}_answers.json")

        with open(img_p) as f:
            imgs = json.load(f)['images']
        with open(q_p) as f:
            qs = json.load(f)['questions']
        with open(a_p) as f:
            ans = json.load(f)['answers']

        q_map = {q['id']: q for q in qs}
        a_map = {a['question_id']: a for a in ans}

        file_to_qa = defaultdict(list)
        for entry in imgs:
            fname = entry['file_name']
            for qid in entry['questions_ids']:
                if qid in q_map and qid in a_map:
                    file_to_qa[fname].append({
                        'question_id': qid,
                        'question': q_map[qid]['question'],
                        'type': q_map[qid]['type'],
                        'answer': a_map[qid]['answer']
                    })

        fnames = sorted(list(file_to_qa.keys()))[:max_files]
        return [{'file_name': f, 'qa_pairs': file_to_qa[f]} for f in fnames]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        fname = item['file_name']

        if self.cache_in_ram and fname in self.image_cache:
            im1, im2, mask1, mask2 = self.image_cache[fname]
        else:
            im1 = cv2.imread(os.path.join(self.im1_dir, fname))
            im2 = cv2.imread(os.path.join(self.im2_dir, fname))
            l1 = cv2.imread(os.path.join(self.label1_dir, fname))
            l2 = cv2.imread(os.path.join(self.label2_dir, fname))

            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            l1 = cv2.cvtColor(l1, cv2.COLOR_BGR2RGB)
            l2 = cv2.cvtColor(l2, cv2.COLOR_BGR2RGB)

            im1 = cv2.resize(im1, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            im2 = cv2.resize(im2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            l1 = cv2.resize(l1, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            l2 = cv2.resize(l2, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            mask1 = rgb_mask_to_class_indices(l1)
            mask2 = rgb_mask_to_class_indices(l2)

        t1_tensor = torch.from_numpy(im1.transpose(2, 0, 1)).float() / 255.0
        t2_tensor = torch.from_numpy(im2.transpose(2, 0, 1)).float() / 255.0
        mask1_tensor = torch.from_numpy(mask1).long()
        mask2_tensor = torch.from_numpy(mask2).long()

        # Remote Sensing Symmetrical Augmentations
        if self.augment:
            # 1. Random Horizontal Flip
            if torch.rand(1) > 0.5:
                t1_tensor = TF.hflip(t1_tensor)
                t2_tensor = TF.hflip(t2_tensor)
                mask1_tensor = TF.hflip(mask1_tensor)
                mask2_tensor = TF.hflip(mask2_tensor)
            # 2. Random Vertical Flip
            if torch.rand(1) > 0.5:
                t1_tensor = TF.vflip(t1_tensor)
                t2_tensor = TF.vflip(t2_tensor)
                mask1_tensor = TF.vflip(mask1_tensor)
                mask2_tensor = TF.vflip(mask2_tensor)
            # 3. Random Orthogonal Rotations (0, 90, 180, 270 deg)
            rot_k = torch.randint(0, 4, (1,)).item()
            if rot_k > 0:
                t1_tensor = torch.rot90(t1_tensor, rot_k, [1, 2])
                t2_tensor = torch.rot90(t2_tensor, rot_k, [1, 2])
                mask1_tensor = torch.rot90(mask1_tensor, rot_k, [0, 1])
                mask2_tensor = torch.rot90(mask2_tensor, rot_k, [0, 1])

        # ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t1_tensor = (t1_tensor - mean) / std
        t2_tensor = (t2_tensor - mean) / std

        q_tokens = self.tokenizer.encode(item['question'])
        ans_target = torch.tensor(item['ans_idx'], dtype=torch.long)

        return {
            'file_name': fname,
            't1': t1_tensor,
            't2': t2_tensor,
            'mask1': mask1_tensor,
            'mask2': mask2_tensor,
            'question_tokens': q_tokens,
            'question_text': item['question'],
            'ans_target': ans_target,
            'answer_text': item['answer']
        }
