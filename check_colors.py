import json
import cv2
import numpy as np

with open('C:/satquery/CDVQA-main/Train_questions.json') as f:
    questions = json.load(f)['questions']
with open('C:/satquery/CDVQA-main/Train_answers.json') as f:
    answers = json.load(f)['answers']
with open('C:/satquery/CDVQA-main/Train_images.json') as f:
    images = json.load(f)['images']

q_map = {q['id']: q for q in questions}
a_map = {a['question_id']: a for a in answers}

# Let's inspect answer vocabulary
all_answers = set(a['answer'] for a in answers)
print("Unique answers count in Train:", len(all_answers))
print("Sample answers:", list(all_answers)[:20])

# Map colors to classes by checking which classes are present when colors are present
classes = [
    "non-vegetated ground surface",
    "tree",
    "low vegetation",
    "water",
    "buildings",
    "playground"
]

color_counts = {}

# Check 50 images
for img_info in images[:100]:
    fname = img_info['file_name']
    path1 = f"C:/satquery/label1/{fname}"
    if not cv2.os.path.exists(path1):
        continue
    l1 = cv2.imread(path1)
    rgbs = set(tuple(c[::-1]) for c in np.unique(l1.reshape(-1, 3), axis=0))
    
    # questions for this img
    for qid in img_info['questions_ids']:
        q_text = q_map[qid]['question'].lower()
        ans = a_map[qid]['answer'].lower()
        for cls in classes:
            if cls in q_text:
                # If question asks if this class changed in first image or changed
                # Let's see
                pass

print("Done checking.")
