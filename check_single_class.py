import json
import cv2
import numpy as np
from collections import defaultdict

with open('C:/satquery/CDVQA-main/Train_questions.json') as f:
    questions = json.load(f)['questions']
with open('C:/satquery/CDVQA-main/Train_answers.json') as f:
    answers = json.load(f)['answers']
with open('C:/satquery/CDVQA-main/Train_images.json') as f:
    images = json.load(f)['images']

q_map = {q['id']: q for q in questions}
a_map = {a['question_id']: a for a in answers}

class_names = [
    "non-vegetated ground surface",
    "tree",
    "low vegetation",
    "water",
    "building",
    "playground"
]

# Look at images where "changed in the first image?" or "Have the areas/regions of X changed?"
# For image 0:
# non-veg: yes
# tree: no
# low veg: yes
# water: no
# building: yes
# playground: no
# colors in l1: (128, 0, 0), (0, 128, 0), (128, 128, 128), (255, 255, 255)
# 3 non-white colors! Exactly matching the 3 classes that changed: non-veg, low veg, building!

# Let's find images where exactly ONE class changed!
for img_info in images[:500]:
    fname = img_info['file_name']
    q_ids = img_info['questions_ids']
    
    # check first 6 questions (the basic change_or_not questions)
    changed_classes = []
    for qid in q_ids[:6]:
        q = q_map[qid]
        a = a_map[qid]
        if q['type'] == 'change_or_not' and 'first' not in q['question'] and 'second' not in q['question']:
            for c in class_names:
                if c in q['question'].lower():
                    if a['answer'].lower() == 'yes':
                        changed_classes.append(c)
    
    if len(changed_classes) == 1:
        path1 = f"C:/satquery/label1/{fname}"
        if cv2.os.path.exists(path1):
            l1 = cv2.imread(path1)
            rgbs = [tuple(c[::-1]) for c in np.unique(l1.reshape(-1, 3), axis=0) if not (c==[255,255,255]).all()]
            print(f"File {fname}: Single changed class = {changed_classes[0]}, non-white colors in l1 = {rgbs}")
