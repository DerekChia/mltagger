import time
import collections
import numpy
import tensorflow as tf

class MLTEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.cost_sum = 0.0
        self.sentence_count = 0.0
        self.sentence_correct_binary = 0.0
        self.sentence_predicted = 0.0
        self.sentence_correct = 0.0
        self.sentence_total = 0.0

        self.token_ap_sum = []
        self.token_predicted = []
        self.token_correct = []
        self.token_total = []

        self.start_time = time.time()


    def calculate_ap(self, true_labels, predicted_scores):
        assert(len(true_labels) == len(predicted_scores))
        indices = numpy.argsort(numpy.array(predicted_scores))[::-1]
        summed, correct, total = 0.0, 0.0, 0.0
        for index in indices:
            total += 1.0
            if true_labels[index] >= 0.5:
                correct += 1.0
                summed += correct / total
        return (summed / correct) if correct > 0.0 else 0.0


    def append_token_data_for_sentence(self, index, true_labels, token_scores):
        if len(self.token_ap_sum) <= index:
            self.token_ap_sum.append(0.0)
            self.token_predicted.append(0.0)
            self.token_correct.append(0.0)
            self.token_total.append(0.0)

        ap = self.calculate_ap(true_labels, token_scores[:len(true_labels)])
        self.token_ap_sum[index] += ap

        for i in range(len(true_labels)):
            if true_labels[i] >= 0.5:
                self.token_total[index] += 1.0
            if token_scores[i] >= 0.5:
                self.token_predicted[index] += 1.0
            if true_labels[i] >= 0.5 and token_scores[i] >= 0.5:
                self.token_correct[index] += 1.0


    def append_data(self, cost, batch, sentence_scores, token_scores_list):
        assert(len(self.token_ap_sum) == 0 or len(self.token_ap_sum) == len(token_scores_list))
        # Summing up all costs across batches
        self.cost_sum += cost

        # Added
        total_count_interesting_labels = []

        # Evaluate each sentence (i) in batch
        for i in range(len(batch)):
            self.sentence_count += 1.0

            # Ground Truth
            true_labels = [1.0 if batch[i][j][-1] != self.config["default_label"] else 0.0 for j in range(len(batch[i]))]
            count_interesting_labels = numpy.array(true_labels).sum()

            # Added
            # print(count_interesting_labels)
            # print( [batch[i][j][-1] for j in range(len(batch[i]))] )
            # print(true_labels)
            # print( 'Sentence', i, sentence_scores[i], [batch[i][j][0] for j in range(len(batch[i]))] )

            total_count_interesting_labels.append(count_interesting_labels)

            if (count_interesting_labels == 0.0 and sentence_scores[i] < 0.5) or (count_interesting_labels > 0.0 and sentence_scores[i] >= 0.5):
                self.sentence_correct_binary += 1.0

            # For every sentence that is more than 0.5, it is correctly predicted?
            if sentence_scores[i] >= 0.5:
                self.sentence_predicted += 1.0

            if count_interesting_labels > 0.0:
                self.sentence_total += 1.0

            if count_interesting_labels > 0.0 and sentence_scores[i] >= 0.5:
                self.sentence_correct += 1.0

            for k in range(len(token_scores_list)):
                self.append_token_data_for_sentence(k, true_labels, token_scores_list[k][i])

        # print('Interesting Labels', total_count_interesting_labels)


    # def get_results(self, name):
    #     p = (float(self.sentence_correct) / float(self.sentence_predicted)) if (self.sentence_predicted > 0.0) else 0.0
    #     r = (float(self.sentence_correct) / float(self.sentence_total)) if (self.sentence_total > 0.0) else 0.0
    #     f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
    #     f05 = ((1.0 + 0.5*0.5) * p * r / ((0.5*0.5 * p) + r)) if (((0.5*0.5 * p) + r) > 0.0) else 0.0

    #     results = collections.OrderedDict()
    #     results[name + "_cost_sum"] = self.cost_sum
    #     results[name + "_cost_avg"] = self.cost_sum / float(self.sentence_count)
    #     results[name + "_sent_count"] = self.sentence_count
    #     results[name + "_sent_predicted"] = self.sentence_predicted
    #     results[name + "_sent_correct"] = self.sentence_correct
    #     results[name + "_sent_total"] = self.sentence_total
    #     results[name + "_sent_p"] = p
    #     results[name + "_sent_r"] = r
    #     results[name + "_sent_f"] = f
    #     results[name + "_sent_f05"] = f05
    #     results[name + "_sent_correct_binary"] = self.sentence_correct_binary
    #     results[name + "_sent_accuracy_binary"] = self.sentence_correct_binary / float(self.sentence_count)

    #     for k in range(len(self.token_ap_sum)):
    #         mean_ap = self.token_ap_sum[k] / self.sentence_total # only calculating MAP over sentences that have any positive tokens
    #         p = (float(self.token_correct[k]) / float(self.token_predicted[k])) if (self.token_predicted[k] > 0.0) else 0.0
    #         r = (float(self.token_correct[k]) / float(self.token_total[k])) if (self.token_total[k] > 0.0) else 0.0
    #         f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
    #         f05 = ((1.0 + 0.5*0.5) * p * r / ((0.5*0.5 * p) + r)) if (((0.5*0.5 * p) + r) > 0.0) else 0.0

    #         results[name + "_tok_"+str(k)+"_map"] = mean_ap
    #         results[name + "_tok_"+str(k)+"_p"] = p
    #         results[name + "_tok_"+str(k)+"_r"] = r
    #         results[name + "_tok_"+str(k)+"_f"] = f
    #         results[name + "_tok_"+str(k)+"_f05"] = f05

    #     results[name + "_time"] = float(time.time()) - float(self.start_time)

    #     return results

    def get_results(self, name):
        precision = (float(self.sentence_correct) / float(self.sentence_predicted)) if (self.sentence_predicted > 0) else 0.0
        recall = (float(self.sentence_correct) / float(self.sentence_predicted) if (self.sentence_predicted > 0) else 0.0)
        # https://en.wikipedia.org/wiki/F1_score
        f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall > 0.0) else 0.0
        # f0.5 score weighs recall lower than precision (by attenuating the influence of false negatives)
        f05_score = ((1.0 + 0.5 * 0.5) * (precision * recall) / (0.5 * 0.5 * precision + recall)) if (precision + recall > 0.0) else 0.0

        results = collections.OrderedDict()
        # Average cost of each word generated in this Epoch
        results[name + '_cost_avg'] = self.cost_sum / float(self.sentence_count)
        # Total cost generated in this Epoch by summing all loss across each word
        results[name + '_cost_sum'] = self.cost_sum

        results[name + "_sentence_count"] = self.sentence_count
        results[name + "_sentence_predicted"] = self.sentence_predicted
        results[name + "_sentence_correct"] = self.sentence_correct
        results[name + "_sentence_total"] = self.sentence_total
        # Precision
        results[name + '_sentence_precision'] = precision
        # Recall
        results[name + '_sentence_recall'] = recall
        # F1 Score
        results[name + '_sentence_f1_score'] = f1_score
        tf.summary.scalar('sentence_f1_score', f1_score)
        # F05 Score
        results[name + '_sentence_f05_score'] = f05_score
        
        results[name + "_sentence_correct_binary"] = self.sentence_correct_binary
        results[name + "_sentence_accuracy_binary"] = self.sentence_correct_binary / float(self.sentence_count)

        for k in range(len(self.token_ap_sum)):
            mean_ap = self.token_ap_sum[k] / self.sentence_total # only calculating MAP over sentences that have any positive tokens
            p = (float(self.token_correct[k]) / float(self.token_predicted[k])) if (self.token_predicted[k] > 0.0) else 0.0
            r = (float(self.token_correct[k]) / float(self.token_total[k])) if (self.token_total[k] > 0.0) else 0.0
            f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
            f05 = ((1.0 + 0.5*0.5) * p * r / ((0.5*0.5 * p) + r)) if (((0.5*0.5 * p) + r) > 0.0) else 0.0

            results[name + "_tok_"+str(k)+"_map"] = mean_ap
            results[name + "_tok_"+str(k)+"_p"] = p
            results[name + "_tok_"+str(k)+"_r"] = r
            results[name + "_tok_"+str(k)+"_f"] = f
            results[name + "_tok_"+str(k)+"_f05"] = f05

        results[name + "_time"] = float(time.time()) - float(self.start_time)

        tf.summary.merge_all()
        return results




