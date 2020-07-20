import argparse
import numpy as np
import cv2
from enum import Enum
import math

account_id = {}
fpd = 16.0 # frames per day
fpt = 7.0 # frames per transaction

class AccountType(Enum):
    STUDENT = 1
    EMPLOYEE = 2
    RESTAURANT = 3
    GROCERY_STORE = 4
    APARTMENT = 5
    UNIVERSITY = 6
    COMPANY = 7
    BANK = 8
class Account:
    def __init__(self, name, x, y, acctType):
        self.name = name
        self.x = x
        self.y = y
        self.type = acctType
class Pymt:
    def __init__(self, sender_id, receiver_id, amount, day):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.amount = amount
        self.day = day

def initializeLocations(image_size, margin):
    num_accounts = 10
    account_name = ['student A', 'student B', 'employee A', 'restaurant A', 'restaurant B', 'grocery store', 'rent', 'university', 'company', 'bank']
    account_type = [AccountType.STUDENT, AccountType.STUDENT, AccountType.EMPLOYEE, AccountType.RESTAURANT, AccountType.RESTAURANT, AccountType.GROCERY_STORE, AccountType.APARTMENT, AccountType.UNIVERSITY, AccountType.COMPANY, AccountType.BANK]
    id = 0
    for account in account_name:
        account_id[account] = id
        id += 1
    # student A & B live together
    # - usually go to restaurant A for dinner (~3 times per week $), sometimes restaurant B (~1 time per week $$)
    # - after dinner sometimes they go to the grocery store after eating at A because it's close (~1 time per week)
    # - they get paid by the university at the end of every month (~$2600)
    # - they pay their share of the rent, and credit balance after they get paid by the end of each month
    # employee A lives in the same city as the students
    # - usually go to restaurant B (~3 times per week) more often than A (~1 time per week)
    # - get paid by the company at the end of every month (~$20000)
    # - pay the rent and credit balance after he get paid by the end of each month

    # spread accounts on a circle
    radius = image_size/2 - margin
    center = image_size/2 - 1
    delta_theta = 2*math.pi/num_accounts
    accounts = []
    for i in range(num_accounts):
        accounts.append(Account(account_name[i], int(center+radius*math.cos(delta_theta*i)), int(center-radius*math.sin(delta_theta*i)), account_type[i]))
    return accounts

def resizeImages(images, target_size):
    resized_images = np.ones([images.shape[0], target_size, target_size], np.uint8)*255
    for i in range(len(images)):
        resized_images[i] = cv2.resize(images[i], (target_size, target_size))
    return resized_images

def synthesizeTranscations():
    transactions= []
    # randomly sample which week of the month [0, 1, 2, 3]
    # bill and salary payments happens at 3rd week
    week = np.random.choice([0, 1, 2, 3])

    # Student A & B go to restaurants together (80% time restaurant A, 20% restaurant B)
    restaurantA_prob = np.random.normal(0.8, 0.1)
    restaurantA_prob = max(0.0, restaurantA_prob)
    restaurantA_prob = min(1.0, restaurantA_prob)
    num_restaurantA = int(restaurantA_prob*7)

    rng = np.random.RandomState()
    rand_ids = list(range(7))
    rng.shuffle(rand_ids)
    days_restaurantA = rand_ids[:num_restaurantA]
    days_restaurantB = rand_ids[num_restaurantA:]
    amount_rest_A = 20
    amount_rest_B = 35

    for day in days_restaurantA:
        price = np.random.normal(amount_rest_A, 3)
        transactions.append(Pymt(account_id['student A'], account_id['restaurant A'], price, day))
        transactions.append(Pymt(account_id['student B'], account_id['restaurant A'], price, day))
    for day in days_restaurantB:
        price = np.random.normal(amount_rest_B, 3)
        transactions.append(Pymt(account_id['student A'], account_id['restaurant B'], price, day))
        transactions.append(Pymt(account_id['student B'], account_id['restaurant B'], price, day))

    # Student A & B goes to the grocery store sometimes after dinner at restaurant A, because it's close, and 70% time A pays it, 30% time B pays it, because A eats more in general
    day_grocery = np.random.choice(days_restaurantA) + fpt/fpd
    amount_grocery = 120

    payer = np.random.choice(['student A', 'student B'], size=1, p = (0.7, 0.3))[0]
    transactions.append(Pymt(account_id[payer], account_id['grocery store'], np.random.normal(amount_grocery, 10), day_grocery))

    # Employee A goes to restaurant (A 20%, B 80%)
    restaurantA_prob = np.random.normal(0.2, 0.1)
    restaurantA_prob = max(0.0, restaurantA_prob)
    restaurantA_prob = min(1.0, restaurantA_prob)
    num_restaurantA = int(restaurantA_prob*7)

    rng = np.random.RandomState()
    rand_ids = list(range(7))
    rng.shuffle(rand_ids)
    days_restaurantA = rand_ids[:num_restaurantA]
    days_restaurantB = rand_ids[num_restaurantA:]

    for day in days_restaurantA:
        transactions.append(Pymt(account_id['employee A'], account_id['restaurant A'], np.random.normal(amount_rest_A, 3), day))
    for day in days_restaurantB:
        transactions.append(Pymt(account_id['employee A'], account_id['restaurant B'], np.random.normal(amount_rest_B, 3), day))

    # paycheck & bill week
    if week == 3:
        # salary: friday of last week
        day = 4
        transactions.append(Pymt(account_id['company'], account_id['employee A'], 20000, day))
        transactions.append(Pymt(account_id['university'], account_id['student A'], 2800, day))
        transactions.append(Pymt(account_id['university'], account_id['student B'], 2800, day))
        # bill: saturday/sunday of last week
        day = np.random.choice([5, 6])
        transactions.append(Pymt(account_id['employee A'], account_id['rent'], 2800, day))
        transactions.append(Pymt(account_id['employee A'], account_id['bank'], np.random.normal(2000, 80), day))
        day = np.random.choice([5, 6])
        transactions.append(Pymt(account_id['student A'], account_id['rent'], 700, day))
        transactions.append(Pymt(account_id['student A'], account_id['bank'], np.random.normal(1000, 50), day))
        day = np.random.choice([5, 6])
        transactions.append(Pymt(account_id['student B'], account_id['rent'], 700, day))
        transactions.append(Pymt(account_id['student B'], account_id['bank'], np.random.normal(1000, 50), day))

    return transactions

def draw_accounts(accounts, initial_size):
    base_frame = np.ones([initial_size, initial_size], np.uint8)*255
    thumb_size = 20
    half_width = int(thumb_size/2)
    for i in range(len(accounts)):
        account = accounts[i]
        thumbnail_path = '/home/ubuntu/PYMT/thumbnails/' + str(account.type) + '.png'
        thumbnail = cv2.imread(thumbnail_path, cv2.IMREAD_GRAYSCALE)
        thumbnail = cv2.resize(thumbnail, (thumb_size, thumb_size))
        base_frame[account.y - half_width: account.y + half_width, account.x - half_width:account.x + half_width] = thumbnail
    return base_frame

def initialize_images(seq_len, initial_size, accounts):
    base_frame = draw_accounts(accounts, initial_size)
    images = np.repeat(base_frame[np.newaxis, :, :], seq_len, axis=0)
    return images

def draw_payment(pymt, accounts, images):
    # day: day of the week, 0, 1, 2, 3, 4, 5, 6
    # at test time, given first 4 days, predict the next 6 days
    src = accounts[pymt.sender_id]
    dest = accounts[pymt.receiver_id]
    all_x = np.linspace(src.x, dest.x, int(fpt))
    all_y = np.linspace(src.y, dest.y, int(fpt))
    timestamp = range(int(pymt.day*fpd), int(pymt.day*fpd+fpt))
    radius = int(np.log1p(pymt.amount))

    for x, y, idx in zip(all_x, all_y, timestamp):
        images[idx] = cv2.circle(images[idx], (int(x), int(y)), radius, 0, -1)

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Payments visualization training set generation.',
        description='Visualize synthetic transactions between different accounts. Videos are saved in \
                     an npz file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
    #                     help='Folder where the training set will be saved.')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=7,
                        help='Number of frames per training sequences.')
    parser.add_argument('--frame_size', type=int, metavar='SIZE', default=64,
                        help='Size of generated frames.')
    parser.add_argument('--training_size', type=int, metavar='SIZE', default=100000,
                        help='Size of training set.')
    args = parser.parse_args()

    args.seq_len = int(args.seq_len*fpd)
    initial_size = 128
    accounts = initializeLocations(initial_size, margin=15)

    # for i in range(args.training_size):
    for i in range(10):
        images = initialize_images(args.seq_len, initial_size, accounts)
        transaction_history = synthesizeTranscations()

        for pymt in transaction_history:
            # each transaction:
            images = draw_payment(pymt, accounts, images)

        # images = resizeImages(images, args.frame_size)

        # write visualization to videos
        video_file = '/home/ubuntu/PYMT/%04d.mp4' % (i)
        print(video_file)
        out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.frame_size*2, args.frame_size*2), isColor=False)

        for i in range(args.seq_len):
            out.write(images[i])
        out.release()