import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 intent_size,
                 dropout=0.2):
        # def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):

        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size,
                          enc_hidden_size,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.fc1 = nn.Linear(dec_hidden_size, intent_size)

    def forward(self, x, lengths):

        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]

        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()  # 2, batch_size, 100

        hid = torch.cat((hid[-2], hid[-1]), dim=1)  # hid[-1] 的 size 为 64，100
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)  # 1, 64, 100
        intent_hid = hid.squeeze(0)
        intent = torch.relu(self.fc1(intent_hid))

        # print('intent size = ',intent.size())

        # embedded = self.embed(x)
        # out, hid = self.rnn(embedded)
        # hid = torch.cat([hid[-2], hid[-1]], dim=1)
        # hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        # print('encoder out size = ', out.size())
        # print('encoder hid size = ', hid.size())

        return out, hid, intent


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2,
                                   dec_hidden_size,
                                   bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size,
                                    dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, enc_hidden_size

        batch_size = output.size(0)  # output 是 y , 形状是 batch_size,
        output_len = output.size(1)  #  y 维度1上的长度，即目标句子的长度
        input_len = context.size(1)  # context 是 encoder_output  维度1是 句子的序列长度

        context_in = self.linear_in(context.view(
            batch_size * input_len,
            -1)).view(batch_size, input_len,
                      -1)  # batch_size, output_len, dec_hidden_size
        # context_in = self.linear_in(context) # batch_size, output_len, dec_hidden_size
        # print('context_in size = ', context_in.size())
        # print ('attention output size =', output.size())

        attn = torch.bmm(output, context_in.transpose(
            1, 2))  # batch_size, output_len, context_len

        # print('attn size = ', attn.size())
        # print('mask size =', mask.size())
        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)  # batch_size, output_len, context_len

        context = torch.bmm(attn,
                            context)  # batch_size, output_len, enc_hidden_size

        output = torch.cat((context, output),
                           dim=2)  # batch_size, output_len, hidden_size*2

        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)  # 64，output_len, 100

        # print('attention output size = ',output.size())
        # print('attention attn size = ', attn.size())
        return output, attn


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 enc_hidden_size,
                 dec_hidden_size,
                 dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len,
                              device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len,
                              device=x_len.device)[None, :] < y_len[:, None]
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(
            self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)

        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)
        # print(mask.size())

        # code.interact(local=locals())
        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)
        # print('decoder output size = ', output.size())
        # print('decoder hid size = ', hid.size())
        # print('decoder attn size =', attn.size())
        return output, hid, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid, intent = self.encoder(x, x_lengths)
        output, decoder_hid, attn = self.decoder(ctx=encoder_out,
                                                 ctx_lengths=x_lengths,
                                                 y=y,
                                                 y_lengths=y_lengths,
                                                 hid=hid)
        return output, attn, intent

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid, intent = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(
                ctx=encoder_out,
                ctx_lengths=x_lengths,
                y=y,
                y_lengths=torch.ones(batch_size).long().to(y.device),
                hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)

    def intent_pred(self, x, x_lengths):
        encoder_out, hid, intent = self.encoder(x, x_lengths)
        return intent


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class IntentCriterion(nn.Module):
    def __init__(self):
        super(IntentCriterion, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, intent_input, intent_target):
        # intent_target = torch.LongTensor(intent_target).unsequeeze(1)
        # intent_target = torch.zeros(batch_size, intent_size).scatter(1,intent_target,1)
        output = self.loss(intent_input, intent_target)
        return output
