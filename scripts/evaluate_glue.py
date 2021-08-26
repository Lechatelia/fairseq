import os
import sys
import shutil
import argparse
import torch 
TASKS = ['MNLI', 'QNLI', 'QQP',  'RTE',  'SST-2', 'MRPC', 'CoLA', 'STS-B')
from fairseq.models.roberta import RobertaModel

def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def eval_task(task_name, args):
    
    checkpoint_path = os.path.join(args.checkpoints, task_name, 'checkpoints')
    data_path = os.path.join(args.data_dir, task_name+'-bin')
    roberta = RobertaModel.from_pretrained(
        checkpoint_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=data_path
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open('glue_data/RTE/dev.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            ncorrect += int(prediction_label == target)
            nsamples += 1
    print('| Accuracy: ', float(ncorrect)/float(nsamples))


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, 
                        default='/mnt/lustre/share_data/zhujinguo/data/bert_pretrain_data/glue_data')
    parser.add_argument('--checkpoints', help='directory to save data to', type=str, 
                        default='workdirs/slurm_roberta_large_glue/finetune')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all')
    args = parser.parse_args(arguments)

    tasks = get_tasks(args.tasks)



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))