# -*- coding: utf-8 -*
import torch.nn as nn
from model_pytorch import *
from data_pytorch import *
from my_metrics import *
import os

train_file = './dataset/chinese_data/pytorch_processed_data.text'
dev_file = './dataset/chinese_data/pytorch_processed_data_test.text'

train_input, train_output, train_intent = load_data(train_file)
dev_input, dev_output, dev_intent = load_data(dev_file)

word_vocab = VocabWord2vec()
index2word = word_vocab.id2token
word2index = word_vocab.token2id

slot2index, slot_size = build_dict(train_output + dev_output)
index2slot = {v: k for k, v in slot2index.items()}

intent2index, intent_size = build_dict(train_intent + dev_intent)
index2intent = {v: k for k, v in intent2index.items()}

train_input, train_output, train_intent = encode(train_input, train_output,
                                                 train_intent, word2index,
                                                 slot2index, intent2index)
dev_input, dev_output, dev_intent = encode(dev_input, dev_output, dev_intent,
                                           word2index, slot2index,
                                           intent2index)

batch_size = 64
train_data = gen_examples(train_input, train_output, train_intent, batch_size)
# random.shuffle(train_data)
dev_data = gen_examples(dev_input, dev_output, dev_intent, batch_size)

word_vocab = VocabWord2vec()
embeddings = word_vocab.embeddings
vocab_size = word_vocab.vocab_size

en_vocab_size = vocab_size
cn_vocab_size = slot_size
embed_size = 150
hidden_size = 100
dropout = 0.2

# model
encoder = Encoder(vocab_size=en_vocab_size,
                  embed_size=embed_size,
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size,
                  intent_size=intent_size,
                  dropout=dropout)

decoder = Decoder(vocab_size=cn_vocab_size,
                  embed_size=embed_size,
                  enc_hidden_size=hidden_size,
                  dec_hidden_size=hidden_size,
                  dropout=dropout)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(encoder, decoder)
model = model.to(device)

model_dir = os.path.dirname(__file__) + '/pytorch_model_dir'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

crit = LanguageModelCriterion().to(device)
intent_crit = IntentCriterion().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())


def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        slot_accs = []
        intent_accs = []
        for it, (mb_x, mb_x_lengths, mb_y, mb_y_lengths,
                 mb_intent) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).long().to(device)

            mb_x_lengths = torch.from_numpy(mb_x_lengths).long().to(device)
            mb_input = torch.from_numpy(mb_y[:, :-1]).long().to(device)
            mb_out = torch.from_numpy(mb_y[:, 1:]).long().to(device)
            mb_y_lengths = torch.from_numpy(mb_y_lengths - 1).long().to(device)
            mb_y_lengths[mb_y_lengths <= 0] = 1

            mb_intent = torch.from_numpy(mb_intent).long().squeeze().to(device)

            mb_pred, attn, intent_pred = model(mb_x, mb_x_lengths, mb_input,
                                               mb_y_lengths)

            mb_out_mask = torch.arange(
                mb_y_lengths.max().item(),
                device=device)[None, :] < mb_y_lengths[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = crit(mb_pred, mb_out, mb_out_mask) + intent_crit(
                intent_pred, mb_intent)

            num_words = torch.sum(mb_y_lengths).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            mb_pred = np.argmax(mb_pred.cpu().detach().numpy(), axis=2)
            mb_x_lengths = mb_x_lengths.cpu().numpy()
            mb_out = mb_out.cpu().numpy()
            mb_intent = mb_intent.cpu().numpy()
            intent_pred = np.argmax(intent_pred.cpu().detach().numpy(), axis=1)

            if it == 0:
                index = random.choice(range(len(mb_x)))
                # mb_x_lengths = mb_x_lengths.cpu().numpy()
                sen_len = mb_x_lengths[index]
                mb_x = mb_x.cpu().numpy()
                # print('mb_y :',mb_y)
                # mb_pred = np.argmax(mb_pred.cpu().detach().numpy(),axis=2)
                print('mb_pred shape:', mb_pred.shape)
                # mb_intent = mb_intent.cpu().numpy()
                # intent_pred = np.argmax(intent_pred.cpu().detach().numpy(),axis =1)

                print('Input sentence :',
                      index_seq2word(mb_x[index], index2word)[:sen_len])
                print('Slot Truth :',
                      index_seq2slot(mb_out[index], index2slot)[:sen_len - 1])
                print('Slot Prediction:',
                      index_seq2slot(mb_pred[index], index2slot)[:sen_len - 1])
                print('Intent Truth :', index2intent[mb_intent[index]])
                print('Intent Prediction :', index2intent[intent_pred[index]])

            true_slot = mb_out
            true_length = list(map(lambda x: x - 1, mb_x_lengths))
            decoder_prediction = mb_pred

            slot_acc = accuracy_score(true_slot, decoder_prediction,
                                      true_length)
            slot_accs.append(slot_acc)
            intent_acc = accuracy_score(mb_intent, intent_pred)
            intent_accs.append(intent_acc)

    print("evaluation loss", total_loss / total_num_words)
    print('slot acc =', np.average(slot_accs))
    print('intent acc =', np.average(intent_accs))


modelpath = model_dir + '/seq2seq_model.pth'


def train(model, data, num_epochs=30):

    if os.path.exists(modelpath):
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['seq2seq'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(num_epochs):
        total_num_words = total_loss = 0.
        model.train()

        for it, (mb_x, mb_x_lengths, mb_y, mb_y_lengths,
                 mb_intent) in enumerate(data):

            mb_x = torch.from_numpy(mb_x).long().to(device)  # 输入句子
            mb_x_lengths = torch.from_numpy(mb_x_lengths).long().to(
                device)  # 输入句子实际长度
            mb_input = torch.from_numpy(mb_y[:, :-1]).long().to(
                device)  # 在训练时，decoder的输入
            mb_out = torch.from_numpy(mb_y[:, 1:]).long().to(
                device)  # decoder的实际标注
            mb_y_lengths = torch.from_numpy(mb_y_lengths - 1).long().to(
                device)  # decoder 序列的长度，在这里是和句子长度一致，在翻译任务中往往不一致
            mb_y_lengths[mb_y_lengths <= 0] = 1

            mb_intent = torch.from_numpy(mb_intent).long().squeeze().to(device)

            mb_pred, attn, intent_pred = model(mb_x, mb_x_lengths, mb_input,
                                               mb_y_lengths)

            mb_out_mask = torch.arange(
                mb_y_lengths.max().item(),
                device=device)[None, :] < mb_y_lengths[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = crit(mb_pred, mb_out, mb_out_mask) + intent_crit(
                intent_pred, mb_intent)
            num_words = torch.sum(mb_y_lengths).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if it % 10 == 0:
                print("epoch", epoch, "iteration", it, "loss", loss.item())

        print("epoch", epoch, "training loss", total_loss / total_num_words)

        if epoch % 5 == 0:
            print("evaluating on dev...")
            evaluate(model, dev_data)
            state = {
                'seq2seq': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, model_dir + '/seq2seq_model.pth')


train(model, train_data, num_epochs=5)


def translate_dev(i):
    model.eval()

    en_sent = " ".join([index2word[word] for word in dev_input[i][1:-1]])
    print(en_sent)
    print(" ".join([index2slot[word] for word in dev_output[i][1:-1]]))

    # sent = nltk.word_tokenize(en_sent.lower())

    bos = torch.Tensor([[slot2index["<BOS>"]]]).long().to(device)
    # mb_x = torch.Tensor([[word2index.get(w, 0) for w in sent]]).long().to(device)

    mb_x = [dev_input[i]]
    mb_x = torch.Tensor(mb_x).long().to(device)
    mb_x_len = torch.Tensor([len(dev_input[i])]).long().to(device)

    translation, attention = model.translate(mb_x, mb_x_len, bos)

    translation = [index2slot[i] for i in translation.data.cpu().numpy().reshape(-1)]

    trans = []
    for word in translation:
        if word != "<EOS>":
            trans.append(word)
        else:
            break
    print(" ".join(trans))
    intent_pred = model.intent_pred(mb_x, mb_x_len)
    intent = np.argmax(intent_pred.cpu().detach().numpy(), axis=1)
    intent = [index2intent[i] for i in intent]
    true_intent = [index2intent[i] for i in dev_intent[i]]
    print('Truth intent =', true_intent)
    print('predict intent = ', intent)


for i in range(90, 100):
    translate_dev(i)
    print()
