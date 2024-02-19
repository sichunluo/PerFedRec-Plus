from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set,valid_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set,valid_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set,valid_set)

        self.model_name_ = (conf.__getitem__('model.name'))
        self.embedding_size = (conf.__getitem__('embedding.size'))

        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.msg = f'Emb size: {self.embedding_size}\n'


    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def predict_local(self, u):
        pass

    def predict_cluster(self, u):
        pass



    
    def test(self,test_mode='valid', model_type='global_model'):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        rec_list = {}
        
        if test_mode=='test':
            user_count = len(self.data.test_set)
            for i, user in enumerate(self.data.test_set):
                if model_type=='global_model':
                    candidates = self.predict(user)
                elif model_type=='local_model':
                    candidates = self.predict_local(user)
                    if candidates is None:
                        candidates = self.predict(user)
                elif model_type=='cluster_model':
                    candidates = self.predict_cluster(user)
                    if candidates is None:
                        candidates = self.predict(user)
                elif model_type=='mix':
                    candidates = self.predict_local(user)
                    if candidates is None:
                        candidates = self.predict(user)
                    candidates2 = self.predict_cluster(user)
                    if candidates2 is None:
                        candidates2 = self.predict(user)
       
                    candidates = candidates + self.predict(user) + candidates2

                rated_list, li = self.data.user_rated(user)
                for item in rated_list:
                    candidates[self.data.item[item]] = -10e8
                ids, scores = find_k_largest(self.max_N, candidates)
                item_names = [self.data.id2item[iid] for iid in ids]
                rec_list[user] = list(zip(item_names, scores))
                if i % 1000 == 0:
                    process_bar(i, user_count)
        else:
            user_count = len(self.data.valid_set)
            for i, user in enumerate(self.data.valid_set):
                if model_type=='global_model':
                    candidates = self.predict(user)
                elif model_type=='local_model':
                    candidates = self.predict_local(user)
                    if candidates is None:
                        candidates = self.predict(user)
                elif model_type=='cluster_model':
                    candidates = self.predict_cluster(user)
                    if candidates is None:
                        candidates = self.predict(user)
                elif model_type=='mix':
                    candidates = self.predict_local(user)
                    if candidates is None:
                        candidates = self.predict(user)
                    candidates = candidates + self.predict(user) * 2
                rated_list, li = self.data.user_rated(user)
                for item in rated_list:
                    candidates[self.data.item[item]] = -10e8
                ids, scores = find_k_largest(self.max_N, candidates)
                item_names = [self.data.id2item[iid] for iid in ids]
                rec_list[user] = list(zip(item_names, scores))
                if i % 1000 == 0:
                    process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list


    def evaluate(self):
        if not 'PerFedRec' in self.model_name_:
            rec_list = self.test(test_mode='test', model_type='global_model')
            current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            out_dir = self.output['-dir']
            file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
            print('The result has been output to ', abspath(out_dir), '.')
            file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
            self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN, print_for_test=True)
            self.model_log.add(f'###Evaluation Results### best')
            self.model_log.add(self.result)
            # print(self.result)
            print_result = self.result + [('\n'+'Best Epoch:'+str(self.best_epoch)+self.msg)]
            # print(print_result)
            FileIO.write_file(out_dir, file_name, print_result)
            print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))
        else:
            rec_list = self.test(test_mode='test', model_type='mix')
            self.msg = self.msg + '\n mix'
            current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            out_dir = self.output['-dir']
            file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
            print('The result has been output to ', abspath(out_dir), '.')
            file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
            self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN, print_for_test=True)
            self.model_log.add(f'###Evaluation Results### best')
            self.model_log.add(self.result)
            print_result = self.result + [('\n'+'Best Epoch:'+str(self.best_epoch)+self.msg)]
            FileIO.write_file(out_dir, file_name, print_result)
            print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))



    def fast_evaluation(self, epoch ,model_type='global_model'):
        print('Evaluating the model...')
        rec_list = self.test('valid', model_type)
        measure = ranking_evaluation(self.data.valid_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
                self.best_epoch = epoch
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
            self.best_epoch = epoch
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure
