import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model


def separate_data(data):
    nRows = len(data)
    data_index = np.random.choice(nRows, nRows//5, replace = False)
    test_data_index = data_index[:nRows//10]
    validation_data_index = data_index[nRows//10:]
    test_data_index = np.array(sorted(test_data_index))
    validation_data_index = np.array(sorted(validation_data_index))
    setRemain = np.array(list(set(np.arange(nRows)) - set(data_index)))
    print(test_data_index.shape, validation_data_index.shape, setRemain.shape)
    return data.ix[test_data_index], data.ix[validation_data_index], data.ix[sorted(setRemain)]



def train(trainingData, validationData, features, responseLabel, method = "vanilla_logit"):
    train_x = trainingData[features].as_matrix()
    train_y = trainingData[responseLabel].as_matrix()[:,0]
    validation_x = validationData[features]
    validation_y = validationData[responseLabel]
    print(train_x)
    if method == "vanilla_logit":
        return vanilla_logit(train_x, train_y, validation_x, validation_y)
    elif method == "bagging_logit":
        return bagging_logit(train_x, train_y, validation_x, validation_y)
       

def vanilla_logit(train_x, train_y, validation_x, validation_y):
    model_lr = sklearn.linear_model.LogisticRegression()
    model_lr.fit(train_x, train_y)
    return model_lr    

def bagging_logit(train_x, train_y, validation_x, validation_y):
    positive_train_x = train_x[train_y > 0]
    negative_train_x = train_x[train_y == 0]
    positive_train_y = train_y[train_y > 0]
    negative_train_y = train_y[train_y == 0]
    model_list = []
    negative_train_number = len(positive_train_x) * 1
    for i in range(50):
        chooseIndex = np.random.choice(np.array(negative_train_x.shape[0]), negative_train_number, replace = False)
        selected_negative_x = negative_train_x[sorted(chooseIndex)]
        selected_negative_y = negative_train_y[sorted(chooseIndex)]
        all_train_data_x = np.vstack((positive_train_x, selected_negative_x))
        all_train_data_y = np.concatenate((positive_train_y, selected_negative_y))
        model_lr = sklearn.linear_model.LogisticRegression()
        model_lr.fit(all_train_data_x, all_train_data_y)
        model_list.append(model_lr)
    return model_list
    

def test(testingData, features, responseLabel, model, method = "vanilla_logit"):
    if method == "vanilla_logit":
        return test_vanilla_logit(testingData, features, responseLabel, model)
    elif method == "bagging_logit":
        return test_bagging_logit(testingData, features, responseLabel, model)


def test_bagging_logit(testingData, features, responseLabel, model):
    test_x = testingData[features]
    test_y = testingData[responseLabel]
    predicted_result_list = []
    for each_model in model:
        predicted_result_list.append(each_model.predict(test_x))
    predict_y = np.mean(np.vstack(predicted_result_list), axis = 0)
    predict_y[predict_y > 0.5] = 1
    predict_y[predict_y <= 0.5] = 0
    test_y_result = test_y.as_matrix()[:,0]
    test_y_result_positive = test_y_result[test_y_result>0]
    print(test_y_result_positive)
    print("False Negative: ")
    print(np.mean(predict_y[test_y_result > 0] == test_y_result_positive))
    print("False Positive: ")
    print(np.mean(predict_y[predict_y > 0] == test_y_result[predict_y > 0]))
    print("True Negative: ")
    print(np.mean(predict_y[predict_y == 0] == test_y_result[predict_y == 0]))
    print("Accuracy: ") 
    print(np.mean(predict_y == test_y_result))





def test_vanilla_logit(testingData, features, responseLabel, model):
    test_x = testingData[features]
    test_y = testingData[responseLabel]
    predict_y = model.predict(test_x)
    print predict_y
    print test_y.as_matrix()[:,0]
    test_y_result = test_y.as_matrix()[:,0]
    test_y_result_positive = test_y_result[test_y_result>0]
    print(test_y_result_positive)
    print("False Negative: ")
    print(np.mean(predict_y[test_y_result > 0] == test_y_result_positive))
    print("False Positive: ")
    print(np.mean(predict_y[predict_y > 0] == test_y_result[predict_y > 0]))
    print("True Negative: ")
    print(np.mean(predict_y[predict_y == 0] == test_y_result[predict_y == 0]))


def main():
    dataFile = pd.read_csv("./aggregatedData.csv")
    features = ['census_tra', 'year', 'hournumber', 'wind_speed', 'drybulb_fahrenheit', 'hourly_precip', 'relative_humidity', 'fz', 'ra', 'ts', 'br', 'sn', 'hz', 'dz', 'pl', 'fg', 'sa', 'up', 'fu', 'sq', 'gs', 'dod_drybulb_fahrenheit', 'month', 'weekday']
    labels = ['shooting_count', 'robbery_count', 'assault_count']
    responseLabel = ['robbery_count']

    testingData, validationData, trainingData = separate_data(dataFile)

    model = train(trainingData, validationData, features, responseLabel, method = "bagging_logit")
    accuracy = test(testingData, features, responseLabel, model, method = "bagging_logit")
    print accuracy


if __name__ == "__main__":
    main() 
