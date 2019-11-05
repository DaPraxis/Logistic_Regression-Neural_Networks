import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


def run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, lambd, isPenalized):
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': lr,
                    'weight_regularization': lambd,
                    'num_iterations': iteration,
                }

    # Logistic regression weights
    weights = np.random.uniform(-0.05,0.05,M+1)
    # weights = np.random.rand(M+1)
    # w = [0.001]*(M+1)
    # weights = np.asarray(w)
    weights = weights.reshape((M+1, 1))
    # weights = np.zeros((M+1, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    # run_check_grad(hyperparameters)

    train_ce = []
    train_frac = []
    valid_ce = []
    valid_frac = []
    test_ce = []
    test_frac = []
    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        if(isPenalized==False):
            f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        else:
            f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        predictions_test = logistic_predict(weights, test_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)
        
        # print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
        #        "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}".format(
        #            t+1, f / N, cross_entropy_train, frac_correct_train*100,
        #            cross_entropy_valid, frac_correct_valid*100))
        # print("ITERATION:{:4d}   TRAIN NLOGL:{:4.2f}   TEST CE{:.6f} TEST FRAC:{:2.2f}".format(
        #     t+1, f/N, cross_entropy_test, frac_correct_test*100)
        # )
        
        
        train_ce.append(cross_entropy_train)
        train_frac.append(frac_correct_train*100)
        valid_ce.append(cross_entropy_valid)
        valid_frac.append(frac_correct_valid*100)
        test_ce.append(cross_entropy_test)
        test_frac.append(frac_correct_test*100)
    # print(">>>>>>>>>")
    # print("smallest TRAIN CE: {:.6f} at iteration {:d} smallest VALID CE: {:.6f} at iteration {:d}".format(min(train_ce), train_ce.index(min(train_ce))+1, min(valid_ce), valid_ce.index(min(valid_ce))+1))
    # x = np.arange(1,hyperparameters['num_iterations']+1, 1)
    # fig, ax = plt.subplots()
    # ax.grid(False)

    
    # ax.plot(x, train_ce, label = "training_ce")
    # ax.plot(x, valid_ce, label = "valid_ce")
    # ax.scatter(train_ce.index(min(train_ce))+1, min(train_ce), label="smallest train_ce")
    # ax.scatter(valid_ce.index(min(valid_ce))+1, min(valid_ce), label="smallest valid_ce")
    # ax.legend(loc='upper right')
    # ax.set(xlabel = "Iteration", ylabel = "Cross Entropy")
    # fig.suptitle("Cross Entropy Error Processing on Dataset "+ training_type +" LR "+str(hyperparameters["learning_rate"])+ " Penalty "+str(hyperparameters["weight_regularization"]), fontsize=9)
    # # plt.show()
    # fig.savefig(training_type+"Learning Rate "+str(hyperparameters["learning_rate"])+ " Penalty "+str(hyperparameters["weight_regularization"])+" Iteration "+str(hyperparameters["num_iterations"])+".png")
    return train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac

def find_best_ce(iteration_max, lr_val, test_numbers, train_inputs, train_targets):
    res = []
    train_min_ce=[]
    valid_min_ce=[]
    train_max_frac = []
    valid_max_frac = []
    ite_max_lst = []
    ite_min_lst = []
    for i in lr_val:
        ite_min = 10000
        ite_max = 0
        sum_ = 0
        sum_2 = 0
        sum_3 = 0
        sum_4 = 0
        for j in range(test_numbers):
            train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac = run_logistic_regression(train_inputs, train_targets, "test", i, iteration_max, 0, False)
            s1 = (min(train_ce), train_ce.index(min(train_ce)))
            s2 = (min(valid_ce), valid_ce.index(min(valid_ce)))
            if(valid_ce.index(min(valid_ce))>ite_max):
                ite_max = valid_ce.index(min(valid_ce))
            elif(valid_ce.index(min(valid_ce))<ite_min):
                ite_min = valid_ce.index(min(valid_ce))
            sum_+=min(valid_ce)
            sum_2+=min(train_ce)
            sum_3+=max(train_frac)
            sum_4+=max(valid_frac)
        avg = sum_/test_numbers
        avg2 = sum_2/test_numbers
        avg3 = sum_3/test_numbers
        avg4 = sum_4/test_numbers
        case = {}
        case["average"] = avg
        case["interval"] = (ite_min, ite_max)
        case["learning_rate"] = i
        res.append(case)
        train_min_ce.append(avg2)
        valid_min_ce.append(avg)
        valid_max_frac.append(avg3)
        train_max_frac.append(avg4)
        ite_max_lst.append(ite_max)
        ite_min_lst.append(ite_min)
        print("with learning_rate {:f} average valid {:.6f} average training {:.6f} interval of iter {:d} - {:d}".format(i, avg, avg2, ite_min, ite_max))
        print("class frac valid {:2.2f}, train {:2.2f}".format(avg4, avg3))
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle("Test for Performance of Model Under Various Learning Rate")
    ax1.plot(lr_val, train_min_ce, label="train_min_ce")
    ax1.plot(lr_val, valid_min_ce, label="valid_min_ce")
    ax1.legend(loc='upper right')
    ax2.plot(lr_val, ite_max_lst, label="max iteration")
    ax2.plot(lr_val, ite_min_lst, label="min iteration")
    ax2.legend(loc='upper right')
    ax3.plot(lr_val, train_max_frac, label="train_max_frac")
    ax3.plot(lr_val, valid_max_frac, label="valid_max_frac")
    ax3.legend(loc='upper right')
    plt.show()
    return res

def find_best_frac(iteration_val, lr_val, test_numbers, train_inputs, train_targets):
    res = []
    train_min_ce=[]
    valid_min_ce=[]
    train_max_frac = []
    valid_max_frac = []
    ite_max_lst = []
    ite_min_lst = []
    for i in iteration_val:
        ite_min = 10000
        ite_max = 0
        sum_ = 0
        sum_2 = 0
        sum_3 = 0
        sum_4 = 0
        for j in range(test_numbers):
            train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac = run_logistic_regression(train_inputs, train_targets, "test", lr_val, i, 0, False)
            s1 = (min(train_ce), train_ce.index(min(train_ce)))
            s2 = (min(valid_ce), valid_ce.index(min(valid_ce)))
            if(valid_ce.index(min(valid_ce))>ite_max):
                ite_max = valid_ce.index(min(valid_ce))
            elif(valid_ce.index(min(valid_ce))<ite_min):
                ite_min = valid_ce.index(min(valid_ce))
            sum_+=min(valid_ce)
            sum_2+=min(train_ce)
            sum_3+=max(train_frac)
            sum_4+=max(valid_frac)
        avg = sum_/test_numbers
        avg2 = sum_2/test_numbers
        avg3 = sum_3/test_numbers
        avg4 = sum_4/test_numbers
        case = {}
        case["average"] = avg
        case["interval"] = (ite_min, ite_max)
        case["learning_rate"] = i
        res.append(case)
        train_min_ce.append(avg2)
        valid_min_ce.append(avg)
        valid_max_frac.append(avg3)
        train_max_frac.append(avg4)
        ite_max_lst.append(ite_max)
        ite_min_lst.append(ite_min)
        print("with learning_rate {:f} average valid {:.6f} average training {:.6f} interval of iter {:d} - {:d}".format(i, avg, avg2, ite_min, ite_max))
        print("class frac valid {:2.2f}, train {:2.2f}".format(avg4, avg3))
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle("Test for Performance of Model Under Various Iterations")
    ax1.plot(iteration_val, train_min_ce, label="train_min_ce")
    ax1.plot(iteration_val, valid_min_ce, label="valid_min_ce")
    ax1.legend(loc='upper right')
    ax2.plot(iteration_val, ite_max_lst, label="max iteration")
    ax2.plot(iteration_val, ite_min_lst, label="min iteration")
    ax2.legend(loc='upper right')
    ax3.plot(iteration_val, train_max_frac, label="train_max_frac")
    ax3.plot(iteration_val, valid_max_frac, label="valid_max_frac")
    ax3.legend(loc='upper right')
    # plt.show()
    return res

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print ("diff =", diff)

if __name__ == '__main__':
    # train_inputs, train_targets = load_train()
    # training_type = "mnist_train"
    # lr = 0.46
    # iteration = 260
    # train_ce, valid_ce, test_ce = run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, 0 ,False)


    # train_inputs, train_targets = load_train_small()
    # training_type = "mnist_train_small"
    # lr = 0.1
    # iteration = 50
    # train_ce, valid_ce, test_ce = run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, 0, False)

    # input()

    lambdh = [0, 0.001, 0.01, 0.1, 1]
    valid_ce_avg = []
    valid_frac_avg=[]
    train_ce_avg=[]
    train_frac_avg=[]
    small_valid_ce_avg = []
    small_valid_frac_avg=[]
    small_train_ce_avg=[]
    small_train_frac_avg=[]
    test_ult_ce = []
    test_ult_frac = []
    test_ult_ces = []
    test_ult_fracs = []
    for i in lambdh:
        cumu_valid_ce = 0
        cumu__valid_frac = 0
        cumu_train_ce = 0
        cumu__train_frac = 0
        lr = 0.001
        iteration = 600
        train_inputs, train_targets = load_train()
        training_type = "penalize mnist_train"
        train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac= run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, i, True)
        test_ult_ce.append(test_ce[-1])
        test_ult_frac.append(test_frac[-1])
        x = np.arange(1,iteration+1, 1)
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.plot(x, train_ce, label = "training_ce")
        ax.plot(x, valid_ce, label = "valid_ce")
        ax.scatter(train_ce.index(min(train_ce))+1, min(train_ce), label="smallest train_ce")
        ax.scatter(valid_ce.index(min(valid_ce))+1, min(valid_ce), label="smallest valid_ce")
        ax.legend(loc='upper right')
        ax.set(xlabel = "Iteration", ylabel = "Cross Entropy")
        fig.suptitle("Cross Entropy Processing on Dataset "+ training_type +" LR "+str(lr)+ " Penalty "+str(i), fontsize=9)
        # plt.show()
        fig.savefig(training_type+"Learning Rate "+str(lr)+ " Penalty "+str(i)+" Iteration "+str(iteration)+" .png")

        x = np.arange(1,iteration+1, 1)
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.plot(x, train_frac, label = "training_frac")
        ax.plot(x, valid_frac, label = "valid_frac")
        ax.scatter(train_frac.index(min(train_frac))+1, min(train_frac), label="smallest train_frac")
        ax.scatter(valid_frac.index(min(valid_frac))+1, min(valid_frac), label="smallest valid_frac")
        ax.legend(loc='upper right')
        ax.set(xlabel = "Iteration", ylabel = "Frac")
        fig.suptitle("Classification Frac Correction Processing on Dataset "+ training_type +"Frac LR "+str(lr)+ " Penalty "+str(i), fontsize=9)
        # plt.show()
        fig.savefig(training_type+"Learning Rate "+str(lr)+ " Penalty "+str(i)+" Iteration "+str(iteration)+" Frac.png")
        cumu__train_frac += train_frac[-1]
        cumu__valid_frac += valid_frac[-1]
        cumu_train_ce += train_ce[-1]
        cumu_valid_ce += valid_ce[-1]
        for j in range(4):
            train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac= run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, i, True)
            cumu__train_frac += train_frac[-1]
            cumu__valid_frac += valid_frac[-1]
            cumu_train_ce += train_ce[-1]
            cumu_valid_ce += valid_ce[-1]
        valid_ce_avg.append(cumu_valid_ce/5)
        valid_frac_avg.append(cumu__valid_frac/5)
        train_ce_avg.append(cumu_train_ce/5)
        train_frac_avg.append(cumu__train_frac/5)


        cumu_valid_ce = 0
        cumu__valid_frac = 0
        cumu_train_ce = 0
        cumu__train_frac = 0
        # lr = 0.007
        # iteration = 700
        train_inputs, train_targets = load_train_small()
        training_type = "penalize mnist_train_small "
        train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac = run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, i, True)
        test_ult_ces.append(test_ce[-1])
        test_ult_fracs.append(test_frac[-1])
        
        x = np.arange(1,iteration+1, 1)
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.plot(x, train_ce, label = "small_training_ce")
        ax.plot(x, valid_ce, label = "small_valid_ce")
        ax.scatter(train_ce.index(min(train_ce))+1, min(train_ce), label="smallest small train_ce")
        ax.scatter(valid_ce.index(min(valid_ce))+1, min(valid_ce), label="smallest small valid_ce")
        ax.legend(loc='upper right')
        ax.set(xlabel = "Iteration", ylabel = "Cross Entropy")
        fig.suptitle("Cross Entropy Processing on Dataset "+ training_type +" LR "+str(lr)+ " Penalty "+str(i), fontsize=9)
        # plt.show()
        fig.savefig(training_type+"Learning Rate "+str(lr)+ " Penalty "+str(i)+" Iteration "+str(iteration)+" .png")

        x = np.arange(1,iteration+1, 1)
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.plot(x, train_frac, label = "small_training_frac")
        ax.plot(x, valid_frac, label = "small_valid_frac")
        ax.scatter(train_frac.index(min(train_frac))+1, min(train_frac), label="smallest small train_frac")
        ax.scatter(valid_frac.index(min(valid_frac))+1, min(valid_frac), label="smallest small valid_frac")
        ax.legend(loc='upper right')
        ax.set(xlabel = "Iteration", ylabel = "Frac")
        fig.suptitle("Classification Frac Correction Processing on Dataset "+ training_type +"Frac LR "+str(lr)+ " Penalty "+str(i), fontsize=9)
        # plt.show()
        fig.savefig(training_type+"Learning Rate "+str(lr)+ " Penalty "+str(i)+" Iteration "+str(iteration)+"Frac.png")
        cumu__train_frac += train_frac[-1]
        cumu__valid_frac += valid_frac[-1]
        cumu_train_ce += train_ce[-1]
        cumu_valid_ce += valid_ce[-1]
        for j in range(4):
            train_ce, valid_ce, test_ce, train_frac, valid_frac, test_frac= run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, i, True)
            cumu__train_frac += train_frac[-1]
            cumu__valid_frac += valid_frac[-1]
            cumu_train_ce += train_ce[-1]
            cumu_valid_ce += valid_ce[-1]
        small_valid_ce_avg.append(cumu_valid_ce/5)
        small_valid_frac_avg.append(cumu__valid_frac/5)
        small_train_ce_avg.append(cumu_train_ce/5)
        small_train_frac_avg.append(cumu__train_frac/5)

    print(test_ult_ce)
    print(test_ult_ces)
    print(test_ult_frac)
    print(test_ult_fracs)
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.plot(lambdh, train_ce_avg, label = "valid_avg_ce")
    ax.plot(lambdh, valid_ce_avg, label = "train_avg_ce")
    ax.legend(loc='upper right')
    ax.set(xlabel = "Lambda", ylabel = "Cross Entropy")
    fig.suptitle("Average Cross Entropy With Various Penalty On mnist_test", fontsize=9)
    # for i, txt in enumerate(valid_ce_avg):
    #     ax.annotate(txt, (lambdh[i], valid_ce_avg[i]))
    # for i, txt in enumerate(train_ce_avg):
    #     ax.annotate(txt, (lambdh[i], train_ce_avg[i]))
    # plt.show()
    fig.savefig("Average Cross Entropy With Various Penalty on large.png")

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.plot(lambdh, small_valid_ce_avg, label = "small_valid_avg_ce")
    ax.plot(lambdh, small_train_ce_avg, label = "small_train_avg_ce")
    ax.legend(loc='upper right')
    ax.set(xlabel = "Lambda", ylabel = "Cross Entropy")
    fig.suptitle("Average Cross Entropy With Various Penalty On mnist_train_small", fontsize=9)
    # for i, txt in enumerate(small_valid_ce_avg):
    #     ax.annotate(txt, (lambdh[i], small_valid_ce_avg[i]))
    # for i, txt in enumerate(train_ce_avg):
    #     ax.annotate(txt, (lambdh[i], small_train_ce_avg[i]))
    # plt.show()
    fig.savefig("Average Cross Entropy With Various Penalty on small.png")

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.plot(lambdh, valid_frac_avg, label = "valid_avg_frac")
    ax.plot(lambdh, train_frac_avg, label = "train_avg_frac")
    ax.legend(loc='upper right')
    ax.set(xlabel = "Lambda", ylabel = "Classification Frac Correction")
    fig.suptitle("Average Classification Frac Correction With Various Penalty On mnist_test", fontsize=9)
    # for i, txt in enumerate(valid_frac_avg):
    #     ax.annotate(txt, (lambdh[i], valid_frac_avg[i]))
    # for i, txt in enumerate(train_frac_avg):
    #     ax.annotate(txt, (lambdh[i], train_frac_avg[i]))
    # plt.show()
    fig.savefig("Average Classification Frac Correction With Various Penalty on large.png")

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.plot(lambdh, small_valid_frac_avg, label = "small_valid_avg_frac")
    ax.plot(lambdh, small_train_frac_avg, label = "small_train_avg_frac")
    ax.legend(loc='upper right')
    ax.set(xlabel = "Lambda", ylabel = "Classification Frac Correction")
    fig.suptitle("Average Classification Frac Correction With Various Penalty On mnist_train_small", fontsize=9)
    # for i, txt in enumerate(valid_frac_avg):
    #     ax.annotate(txt, (lambdh[i], small_valid_frac_avg[i]))
    # for i, txt in enumerate(train_frac_avg):
    #     ax.annotate(txt, (lambdh[i], small_valid_frac_avg[i]))
    # plt.show()
    fig.savefig("Average Classification Frac Correction With Various Penalty on small.png")
    # plt.show()
        


    # lr = 0.30
    # iteration = 400
    # train_inputs, train_targets = load_train()
    # training_type = "penalize mnist_train_test"
    # run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, 0.001, True)

    # lr = 0.006
    # iteration = 400
    # train_inputs, train_targets = load_train_small()
    # training_type = "penalize mnist_train_small_test"
    # run_logistic_regression(train_inputs, train_targets, training_type, lr, iteration, 0.001, True)

    # train_inputs, train_targets = load_train()
    # lr_val = np.arange(0.1, 1.0, 0.1)
    # res = find_best_ce(50, lr_val, 5, train_inputs, train_targets)
    # print(res)

    # train_inputs, train_targets = load_train()
    # lr_val=0.4
    # iteration = np.arange(1, 200, 10)
    # res = find_best_frac(iteration, lr_val, 5, train_inputs, train_targets)
    # print(res)






    
