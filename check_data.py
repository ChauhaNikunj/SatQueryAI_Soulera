import json
import cv2
import numpy as np

# Load questions, answers, images
with open('C:/satquery/CDVQA-main/Train_questions.json') as f:
    questions = json.load(f)['questions']
with open('C:/satquery/CDVQA-main/Train_answers.json') as f:
    answers = json.load(f)['answers']
with open('C:/satquery/CDVQA-main/Train_images.json') as f:
    images = json.load(f)['images']

print(f"Total train images: {len(images)}")
print(f"Total train questions: {len(questions)}")
print(f"Total train answers: {len(answers)}")

# Look at first image
img0 = images[0]
print("First image:", img0)
q_map = {q['id']: q for q in questions if q['id'] in img0['questions_ids']}
a_map = {a['question_id']: a for a in answers if a['question_id'] in img0['questions_ids']}

for qid in img0['questions_ids']:
    print(f"  Q: {q_map[qid]['question']} -> A: {a_map[qid]['answer']}")

l1 = cv2.imread(f"C:/satquery/label1/{img0['file_name']}")
l2 = cv2.imread(f"C:/satquery/label2/{img0['file_name']}")

def get_rgb_unique(bgr_img):
    u = np.unique(bgr_img.reshape(-1, 3), axis=0)
    return [tuple(c[::-1]) for c in u]

print("l1 RGBs:", get_rgb_unique(l1))
print("l2 RGBs:", get_rgb_unique(l2))
