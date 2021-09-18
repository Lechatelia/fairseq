import os
import sys
import shutil
import argparse
import torch 
import numpy as np
TASKS = ['MNLI', 'QNLI', 'QQP',  'RTE',  'SST-2', 'MRPC', 'CoLA', 'STS-B']

from fairseq.models.roberta import RobertaModel
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    elif 'two' in task_names:
        tasks = ['CoLA', 'STS-B']
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def eval_task(task_name, args):
    
    checkpoint_path = os.path.join(args.checkpoints, task_name, 'checkpoints')
    if not os.path.isfile(os.path.join(checkpoint_path, 'checkpoint_best.pt')):
        print('checkpoint not exist! please check!')
        return
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
    preds = []
    labels = []
    if task_name == 'MNLI':
        with open(os.path.join(args.data_dir, task_name,'processed/dev_matched.tsv')) as fin:
            # fin.readline()
            for index, line in enumerate(fin):
                tokens = line.strip().split('\t')
               
                sent1, sent2, target = tokens[8], tokens[9], tokens[15]
                tokens = roberta.encode(sent1, sent2)

                prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                prediction_label = label_fn(prediction)
                ncorrect += int(prediction_label == target)
                preds.append(prediction)
                labels.append(roberta.task.label_dictionary.index(target)-roberta.task.label_dictionary.nspecial)
                nsamples += 1     
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
        preds = np.array(preds)
        labels = np.array(labels)
        acc = simple_accuracy(preds, labels)
        print('Task MNLI-m | accuracy: {} '.format( acc))
        
        with open(os.path.join(args.data_dir, task_name,'processed/dev_mismatched.tsv')) as fin:
            # fin.readline()
            for index, line in enumerate(fin):
                tokens = line.strip().split('\t')
               
                sent1, sent2, target = tokens[8], tokens[9], tokens[15]
                tokens = roberta.encode(sent1, sent2)

                prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                prediction_label = label_fn(prediction)
                ncorrect += int(prediction_label == target)
                preds.append(prediction)
                labels.append(roberta.task.label_dictionary.index(target)-roberta.task.label_dictionary.nspecial)
                nsamples += 1     
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
        preds = np.array(preds)
        labels = np.array(labels)
        acc = simple_accuracy(preds, labels)
        print('Task MNLI-mm  | accuracy: {} '.format( acc))
        return 
                
    if task_name in ['MNLI', 'QNLI', 'QQP',  'RTE',  'SST-2', 'MRPC', 'CoLA', 'STS-B']:
        path_name = os.path.join(args.data_dir, task_name,'processed/dev.tsv') 
    else:
        raise NotImplementedError
    
    with open(path_name) as fin:
        # fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            if task_name == 'QNLI':
                sent1, sent2, target = tokens[1], tokens[2], tokens[3]
                tokens = roberta.encode(sent1, sent2)
            elif task_name == 'QQP':
                sent1, sent2, target = tokens[3], tokens[4], tokens[5]
                tokens = roberta.encode(sent1, sent2)
            elif task_name == 'RTE':
                sent1, sent2, target = tokens[1], tokens[2], tokens[3]
                tokens = roberta.encode(sent1, sent2)
            elif task_name == 'SST-2':
                sent1, target = tokens[0], tokens[1]
                tokens = roberta.encode(sent1)
            elif task_name == 'MRPC':
                sent1, sent2, target = tokens[3], tokens[4], tokens[0]
                tokens = roberta.encode(sent1, sent2)
            elif task_name == 'CoLA':
                sent1,  target = tokens[3], tokens[1]
                tokens = roberta.encode(sent1)
            elif task_name == 'STS-B':
                sent1, sent2, target = tokens[7], tokens[8], tokens[9]
                tokens = roberta.encode(sent1, sent2)
            
            else:
                raise NotImplementedError
            if task_name != 'STS-B':
                prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                prediction_label = label_fn(prediction)
                ncorrect += int(prediction_label == target)
                preds.append(prediction)
                labels.append(roberta.task.label_dictionary.index(target)-roberta.task.label_dictionary.nspecial)
            else:
                # regression task
                preds.append(roberta.predict('sentence_classification_head', tokens,return_logits=True).item())
                labels.append(float(target)/5.0)
            # nsamples += 1     
        # print('| Accuracy: ', float(ncorrect)/float(nsamples))
        preds = np.array(preds)
        labels = np.array(labels)
        
        if task_name == 'CoLA':
            acc = simple_accuracy(preds, labels)
            matthewscorr = matthews_corrcoef(labels, preds)
            print('Task {} | accuracy: {} '.format(task_name, acc))
            print('Task {} | matthews_corrcoef: {} '.format(task_name, matthewscorr ))
        elif task_name in ['MNLI', 'QNLI', 'RTE', 'SST-2', ]:
            acc = simple_accuracy(preds, labels)
            print('Task {} | accuracy: {} '.format(task_name, acc))
        elif task_name in ['MRPC', 'QQP']:
            acc = simple_accuracy(preds, labels)
            print('Task {} | accuracy: {} '.format(task_name, acc))
            f1 = f1_score(y_true=labels, y_pred=preds)
            print('Task {} | f1 score: {} '.format(task_name, f1))
        elif task_name in ['STS-B']:
            pearson_corr = pearsonr(preds, labels)[0]
            spearman_corr = spearmanr(preds, labels)[0]
            print('Task {} | pearson_corr: {} '.format(task_name, pearson_corr))
            print('Task {} | spearman_corr: {} '.format(task_name, spearman_corr))
        else:
            raise NotImplementedError




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
    
    for task in tasks:
        print('--------------')
        print('GLUE: {}'.format(task))
        eval_task(task, args)
        print('--------------\n')



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
