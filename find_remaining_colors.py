import json, cv2, numpy as np

with open('C:/satquery/CDVQA-main/Train_questions.json') as f:
    questions = json.load(f)['questions']
with open('C:/satquery/CDVQA-main/Train_answers.json') as f:
    answers = json.load(f)['answers']
with open('C:/satquery/CDVQA-main/Train_images.json') as f:
    images = json.load(f)['images']

q_map = {q['id']: q for q in questions}
a_map = {a['question_id']: a for a in answers}

# find an image with tree, water, or playground
for img in images[:300]:
    for qid in img['questions_ids']:
        q = q_map[qid]['question'].lower()
        ans = a_map[qid]['answer'].lower()
        if ('water' in q or 'tree' in q or 'playground' in q) and ans == 'yes' and 'changed' in q:
            fname = img['file_name']
            l1 = cv2.imread(f'C:/satquery/label1/{fname}')
            l2 = cv2.imread(f'C:/satquery/label2/{fname}')
            u1 = [tuple(c[::-1]) for c in np.unique(l1.reshape(-1, 3), axis=0) if not (c==[255,255,255]).all()]
            u2 = [tuple(c[::-1]) for c in np.unique(l2.reshape(-1, 3), axis=0) if not (c==[255,255,255]).all()]
            print(f"{fname} | Q: {q} -> {ans} | Colors L1: {u1} | L2: {u2}")
            break
