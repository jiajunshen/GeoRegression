import pandas as pd
import tensorflow as tf
from load_data import *
import os.path
def create_neighborhood_map():
    # Read neighborhood file
    neighborhood_map = {}
    neighborhood_file = open("../data/community_neighbor.csv")
    neighbor_list = neighborhood_file.read()
    for i, record in enumerate(neighbor_list.split('\n')):
        current_neighbor = record.split()
        neighborhood_map[i + 1] = []
        for neighbor in current_neighbor[1:]:
            neighborhood_map[i + 1].append(int(neighbor))
    return neighborhood_map

def bucket(value, bucket_size, lower_limit, upper_limit):
    if value <= lower_limit:
        return 0
    elif value >= upper_limit:
        return int((upper_limit - lower_limit) // bucket_size + 1)
    else:
        return int((value - lower_limit) // bucket_size + 1)

def bucket_data_frame(data_for_region):

    for j in range(12):
        data_for_region['month_%d' % (j + 1)] = 0
    for j in range(6):
        data_for_region['weekday_%d' % j] = 0
    for j in range(24):
        data_for_region['hournumber_%d' %j] = 0
    for j in range(10):
        data_for_region['wind_speed_%d' %j] = 0
    for j in range(10):
        data_for_region['drybulb_fahrenheit_%d' %j] = 0
    for j in range(10):
        data_for_region['dod_drybulb_fahrenheit_%d' %j] = 0
    for j in range(10):
        data_for_region['hourly_precip_%d' %j] = 0
    for j in range(12):
        data_for_region['relative_humidity_%d' %j] = 0

    for index, row in data_for_region.iterrows():
        month_number = row['month']
        data_for_region.set_value(index, 'month_%d' %month_number, 1)

        week_day = row['weekday']
        data_for_region.set_value(index, 'weekday_%d' %week_day, 1)

        hour_number = row['hournumber']
        data_for_region.set_value(index, 'hournumber_%d' %hour_number, 1)

        wind_speed = row['wind_speed']
        wind_speed_bucket = bucket(wind_speed, 5, 0, 40) # 10 buckets
        data_for_region.set_value(index, 'wind_speed_%d' %wind_speed_bucket, 1)

        drybulb_fahrenheit = row['drybulb_fahrenheit']
        drybulb_fahrenheit_bucket = bucket(drybulb_fahrenheit, 15, -20, 100) # 10 buckets
        data_for_region.set_value(index, 'drybulb_fahrenheit_%d' %drybulb_fahrenheit_bucket, 1)

        dod_drybulb_fahrenheit = row['dod_drybulb_fahrenheit']
        dod_drybulb_fahrenheit_bucket = bucket(dod_drybulb_fahrenheit, 10, -40, 40) # 10 buckets
        data_for_region.set_value(index, 'dod_drybulb_fahrenheit_%d' %dod_drybulb_fahrenheit_bucket, 1)

        hourly_precip = row['hourly_precip']
        hourly_precip_bucket = bucket(hourly_precip, 0.1, 0, 0.8) # 10 buckets
        data_for_region.set_value(index, 'hourly_precip_%d' %hourly_precip_bucket, 1)

        relative_humidity = row['relative_humidity']
        relative_humidity_bucket = bucket(relative_humidity, 10, 0, 100) # 12 buckets
        data_for_region.set_value(index, 'relative_humidity_%d' %relative_humidity_bucket, 1)

    return data_for_region


def main():
    import collections
    # Read all data
            
    predictors = ['month_%d' % (j + 1) for j in range(12)] + \
                 ['weekday_%d' % j for j in range(6)] + \
                 ['hournumber_%d' %j for j in range(24)] + \
                 ['wind_speed_%d' %j for j in range(10)] + \
                 ['drybulb_fahrenheit_%d' %j for j in range(10)] + \
                 ['dod_drybulb_fahrenheit_%d' %j for j in range(10)] + \
                 ['hourly_precip_%d' %j for j in range(10)] + \
                 ['relative_humidity_%d' %j for j in range(12)] + \
                 ['fz', 'ra', 'ts', 'br', 'sn', 'hz', 'dz', 'pl', 'fg', 'sa', 'up', 'fu', 'sq', 'gs']

    #targets = ['shooting_count','robbery_count', 'assault_count']
    targets = ['shooting_count']
    num_prediction = 1
    num_feature = len(predictors)
    training_data = []
    training_label = []
    testing_data = []
    testing_label = []
    num_region = 77
    for i in range(1, num_region + 1):
        # Load Data
        print i
        data_for_region = pd.read_csv("../data/processed_%d.csv" % i)
        #data_for_region = bucket_data_frame(data_for_region)
        #data_for_region.to_csv("../data/aggregatedData_%d.csv" % i, index=False)
        training_data.append(np.asarray(data_for_region[data_for_region['year'] < 2013][predictors]))
        training_label.append(np.array(np.asarray(data_for_region[data_for_region['year'] < 2013][targets]) > 0, dtype = np.float32))
        testing_data.append(np.asarray(data_for_region[data_for_region['year'] >= 2013][predictors]))
        testing_label.append(np.array(np.asarray(data_for_region[data_for_region['year'] >= 2013][targets]) > 0, dtype = np.float32))

    train = DataSet(training_data, training_label)
    test = DataSet(testing_data, testing_label)

    Datasets = collections.namedtuple('Datasets', ['train', 'test'])
    all_dataset = Datasets(train = train, test = test)


    x_list = []
    y_list = []
    y_hat_list = []
    loss_list = []
    acc_list = []
    w_list = []
    b_list = []
    all_loss = tf.Variable(0.0, dtype="float32")
    for i in range(num_region):
        x = tf.placeholder("float32", [None, num_feature])
        y_ = tf.placeholder("float32",[None, num_prediction])
        W = tf.Variable(tf.random_normal([num_feature,num_prediction]) / 10.0, dtype = "float32")
        b = tf.Variable(tf.zeros([num_prediction]), dtype = "float32")
        y = tf.sigmoid(tf.add(tf.matmul(x,W), b))
        loss = tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(y) - (1 - y_) * tf.log(1 - y), reduction_indices = 1))
        all_loss = all_loss + loss
        x_list.append(x)
        y_list.append(y_)
        y_hat_list.append(y)
        w_list.append(W)
        b_list.append(b)
        loss_list.append(loss)
    neighborhood_map = create_neighborhood_map() 
    constraint_loss = tf.Variable(0.0, dtype="float32")
    for i in range(num_region):
        neighbors = neighborhood_map[i + 1]
        for neighbor in neighbors:
            constraint_loss += tf.reduce_sum(tf.square(tf.sub(w_list[i], w_list[neighbor - 1])))
    # y_hat_list = tf.pack(y_hat_list)
        
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(tf.reduce_sum(all_loss))
    # Here we only use the sum of the all the losses
    # We will add the loss for penalization later

    all_loss = all_loss + 0.01 * constraint_loss

    train_step = tf.train.AdagradOptimizer(0.001).minimize(all_loss)
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()
    #sess.run(init)
    if os.path.isfile("/home-nfs/jiajun/Research/GeoRegression/code/logistic_model_combined_v2.ckpt"):
        saver.restore(sess, "/home-nfs/jiajun/Research/GeoRegression/code/logistic_model_combined_v2.ckpt")
    else:
        sess.run(init)



    batch_size = 100
    print("before training")
    import time
    previous_time = time.time()
    for i in range(80000):
        #print("time_diff:", time.time() - previous_time)

        batch_train_data, batch_train_label = all_dataset.train.next_balanced_batch(batch_size = 100, neg_ratio = 4.0)
        feed_dict = {}
        for j in range(num_region):
            feed_dict[x_list[j]] = batch_train_data[j]
            feed_dict[y_list[j]] = batch_train_label[j].reshape((-1,1))
            
        _, loss_value, constraint_loss_value = sess.run([train_step, all_loss, constraint_loss], feed_dict=feed_dict)
        
        if i % 5000 == 0:
            print("Step %d" %i)
            print("loss", loss_value)
            print("constraint_loss", constraint_loss_value)
            print("Evaluating Positive...")
            feed_dict_evaluate = {}
            for j in range(num_region):
                feed_dict_evaluate[x_list[j]] = all_dataset.test._positive_data[j]
                y_hat_value = sess.run(y_hat_list[j], feed_dict=feed_dict_evaluate)
                print "Region_%d Positive Accuracy: " %j, np.mean(y_hat_value > 0.25)
            print("Evaluating All...")
            feed_dict_new = {}
            for j in range(num_region):
                feed_dict_new[x_list[j]] = all_dataset.test._data[j]
                y_hat_value = sess.run(y_hat_list[j], feed_dict=feed_dict_new)
                print "Region_%d All Accuracy: " %j, np.mean((y_hat_value > 0.25) == all_dataset.test._labels[j])
            print("=======================================================================================")
        if i % 5000 == 0:
            save_path = saver.save(sess, "/home-nfs/jiajun/Research/GeoRegression/code/logistic_model_combined_v2.ckpt")


if __name__ == "__main__":
    main()
