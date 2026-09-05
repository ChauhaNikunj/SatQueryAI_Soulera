import json

with open('C:/satquery/CDVQA-main/Train_questions.json') as f:
    questions = json.load(f)['questions']
with open('C:/satquery/CDVQA-main/Train_answers.json') as f:
    answers = json.load(f)['answers']
with open('C:/satquery/CDVQA-main/Train_images.json') as f:
    images = json.load(f)['images']

q_map = {q['id']: q for q in questions}
a_map = {a['question_id']: a for a in answers}

# find an image with diverse question types
for img in images[:10]:
    print(f"=== Image {img['id']} ({img['file_name']}) ===")
    for qid in img['questions_ids']:
        print(f"[{q_map[qid]['type']}] Q: {q_map[qid]['question']} -> A: {a_map[qid]['answer']}")
