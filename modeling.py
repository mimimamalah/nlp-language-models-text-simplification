import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter
################################################
##       Part2 --- Language Modeling          ##
################################################   
class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())
            self.embedding.weight.requires_grad = not freeze_embeddings
        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # TODO: Define **bi-directional** LSTM layer with `num_layers` and `dropout_rate`.
        ## Hint: what is the input/output dimensions? how to set the bi-directional model?
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        # TODO: Define the feedforward layer with `num_layers` and `dropout_rate`.
        ## Hint: what is the input/output dimensions for a bi-directional LSTM model?
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # times 2 for bidirectional

    def forward(self, input_id):
        embedding = self.dropout(self.embedding(input_id))
        
        # TODO: Get output from (LSTM layer-->dropout layer-->feedforward layer)
        ## You can add several lines of code here.
 
        lstm_out, _ = self.lstm(embedding)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)

        return output

def train_lstm(model, train_loader, optimizer, criterion, device="cuda:0", tensorboard_path="./tensorboard"):
    """
    Main training pipeline. Implement the following:
    - pass inputs to the model
    - compute loss
    - perform forward/backward pass.
    """
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    running_loss = 0
    epoch_loss = 0
    for i, data in enumerate(tqdm(train_loader)):
        # get the inputs
        inputs = data.to(device)
        
        # TODO: get the language modelling labels form inputs
        labels = data[:, 1:].to(device)

        # to pass test_lstm_pretrained(model) 
        model = model.to(device)
        
        # TODO: Implement forward pass. Compute predicted y by passing x to the model
        y_pred = model(inputs)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)

        # TODO: Compute loss
        loss = criterion(y_pred, labels)

        # TODO: Implement Backward pass. 
        # Hint: remember to zero gradients after each update. 
        # You can add several lines of code here.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i>0 and i % 500 == 0. :
            print(f'[Step {i + 1:5d}] loss: {running_loss / 500:.3f}')
            tb_writer.add_scalar("lstm/train/loss", running_loss / 500, i)
            running_loss = 0.0

    tb_writer.flush()
    tb_writer.close()
    print(f'Epoch Loss: {(epoch_loss / len(train_loader)):.4f}')
    return epoch_loss / len(train_loader)

def test_lstm(model, test_loader, criterion, device="cuda:0"):
    """
    Main testing pipeline. Implement the following:
    - get model predictions
    - compute loss
    - compute perplexity.
    """

    # Testing loop
    batch_loss = 0

    model.eval()
    for data in tqdm(test_loader):
        # get the inputs
        inputs = data.to(device)
        labels = data[:, 1:].to(device)

        # TODO: Run forward pass to get model prediction.
        y_pred = model(inputs)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)
        
        # TODO: Compute loss
        loss = criterion(y_pred, labels)

        batch_loss += loss.item()

    test_loss = batch_loss / len(test_loader)
    
    # TODO: Get test perplexity using `test_loss``
    perplexity = torch.exp(torch.tensor(test_loss))
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test Perplexity: {perplexity:.3f}')
    return test_loss, perplexity

################################################
##       Part3 --- Finetuning          ##
################################################ 

class Encoder(nn.Module):
    def __init__(self, pretrained_encoder, hidden_size):
        super(Encoder, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.hidden_size = hidden_size

    def forward(self, input_ids, input_mask):
        # TODO: Implement forward pass.
        # Hint 1: You should take into account the fact that pretrained encoder is bidirectional.
        # Hint 2: Check out the LSTM docs (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        # Hint 3: Do we need all the components of the pretrained encoder?
        
        encoder_outputs =  self.pretrained_encoder.embedding(input_ids)
        
        encoder_outputs = self.pretrained_encoder.dropout(encoder_outputs)
        
        encoder_outputs, (encoder_hidden, _) = self.pretrained_encoder.lstm(encoder_outputs)
        
        return encoder_outputs, encoder_hidden

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.query_weights = nn.Linear(hidden_size, hidden_size)
        self.value_weights = nn.Linear(hidden_size, hidden_size)
        self.combined_weights = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code
        # permuted to resolve matching issues from Ed-discussion understanding
        weights_query = self.query_weights(query.permute(1, 0, 2))
        weights_values = self.value_weights(values)
        combined = torch.tanh(weights_query+ weights_values)

        attention_scores = self.combined_weights(combined).squeeze(-1)

        attention_scores = attention_scores.masked_fill(mask == 0,  float('-inf'))

        # Attention weights
        weights = F.softmax(attention_scores, dim=-1)

        # The context vector is the weighted sum of the values.
        # completed this line using chatGPT
        # Saw the bmm definition and concluded this is what was asked
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)

        return context, weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, bos_token_id, dropout_rate=0.15, encoder_embedding=None):
        super(Decoder, self).__init__()
        # Note: feel free to change the architecture of the decoder if you like
        if encoder_embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = encoder_embedding
        self.attention = AdditiveAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size, output_size)
        self.bos_token_id = bos_token_id

        self.linear = nn.Linear(2 * hidden_size, hidden_size)

        # On Ed discussion to change the structure of Decoder 
        self.linear_hidden = nn.Linear(4 * hidden_size, hidden_size)


    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensors=None, device=0):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code
        # Hint: Use target_tensors to handle training and inference appropriately

        batch_size = encoder_outputs.size(0)

        max_length = encoder_outputs.size(1)
        
        decoder_outputs = []
        decoder_input = torch.tensor([self.bos_token_id] * batch_size, dtype=torch.long, device=device).unsqueeze(1) 
        
        # Asked chatGPT to give strategies to adjust indexing if your encoder_hidden is structured differently
        # Output the next three methods : 

        #last layer concatenation : did not work
        #decoder_hidden = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1).unsqueeze(0)

        #Select and Reshape : did not work
        #decoder_hidden = encoder_hidden[-2,:,:].unsqueeze(0)

        # Averaging Hidden States : kind of worked
        # Averaging the last forward and backward hidden states
        # decoder_hidden = torch.mean(encoder_hidden[-2:,:,:], dim=0, keepdim=True)

        # The last method kind of worked but predicted mostly unk
        # Thus I asked again by considering that I changed the Decoder structure where I added the hidden linear layer
        # Output :
        # decoder_hidden = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1).unsqueeze(0)
        # And suggested to use this layer inside the loop between  self-gru ad self.out
        # I instead used the linear transformation of the encoder hidden states before passing them to the decoder
        # Reason : This can help in better shaping the information from the encoder to match the requirements of the decoder.
        # I also had batch size mismatch error  
        decoder_hidden = self.linear_hidden(encoder_hidden.view(batch_size, -1)).view(1, batch_size, 100)

        encoder_outputs = self.linear(encoder_outputs)

        for t in range(1, max_length):

            embedded = self.embedding(decoder_input) 
            embedded = self.dropout(embedded)
            
            context, attention_weights = self.attention(decoder_hidden, encoder_outputs, input_mask)
            
            gru_input = torch.cat((embedded, context.unsqueeze(1)), 2)
            
            output, _ = self.gru(gru_input, decoder_hidden)
            
            output = self.out(output.squeeze(1))
            
            decoder_outputs.append(output)

            if target_tensors is not None:
                # Use the actual next token as the next input (teacher forcing)
                decoder_input = target_tensors[:, t].unsqueeze(1)
            else:
                # Use the highest probability token as the next input
                _, top1 = output.topk(1)
                
                # Use chatGPT to implement the next line
                # shares the same underlying data storage as top1 but does not require gradients.
                # Output : top1.unsqueeze(1)
                # Changed to detach : common practice during inference when teacher forcing is not used
                decoder_input = top1.detach()
                    
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        
        # Apply LogSoftmax to the output layer
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden
    
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, bos_token_id, dropout_rate=0.15, pretrained_encoder=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(pretrained_encoder, hidden_size)
        # the embeddings in the encoder and decoder are tied as they're both from the same language
        self.decoder = Decoder(hidden_size, output_vocab_size, bos_token_id, dropout_rate, pretrained_encoder.embedding)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_mask)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden

def seq2seq_eval(model, eval_loader, criterion, device="cuda:0"):
    model.eval()
    model = model.to(device)
    epoch_loss = 0
    
    for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        # TODO: Get the inputs
        input_ids, target_ids, input_mask = data['input_ids'].to(device), data['output_ids'].to(device), data['input_mask'].to(device)

        # TODO: Forward pass
        decoder_outputs, decoder_hidden = model(input_ids, input_mask)

        batch_max_seq_length = decoder_outputs.shape[1]
        labels = target_ids[:, 1:batch_max_seq_length+1].to(device)
        #labels = target_ids[:, 1:].to(device)

        # TODO: Compute loss
        # RunTime error : they suggested to use reshape
        # Previous code : 
        # loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), labels.view(-1)) 
        loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), labels.reshape(-1))
        
        epoch_loss += loss.item()

    model.train()

    return epoch_loss / len(eval_loader)

def seq2seq_train(model, train_loader, eval_loader, optimizer, criterion, num_epochs=20, device="cuda:0", tensorboard_path="./tensorboard"):
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    model = model.to(device)
    best_eval_loss = 1e3 # used to do early stopping
    # I did not use early stopping since I have better results without it

    for epoch in tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=1):
            # TODO: Get the inputs
            input_ids, target_ids, input_mask = data['input_ids'].to(device), data['output_ids'].to(device), data['input_mask'].to(device)
            
            # Forward pass
            decoder_outputs, decoder_hidden = model(input_ids, input_mask)

            batch_max_seq_length = decoder_outputs.shape[1]
            
            labels = target_ids[:, 1:batch_max_seq_length+1].to(device)
            #labels = target_ids[:, 1:].to(device)

            loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), labels.reshape(-1))

            epoch_loss += loss.item()
            
            # TODO: Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99. :    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(train_loader)):.4f}')
        eval_loss = seq2seq_eval(model, eval_loader, criterion, device=device)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        tb_writer.add_scalar("ec-finetune/loss/train", epoch_loss / len(train_loader), epoch)
        tb_writer.add_scalar("ec-finetune/loss/eval", eval_loss, epoch)
        
        # TODO: Perform early stopping based on eval loss
        # Make sure to flush the tensorboard writer and close it before returning
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
        # I removed early stopping because it does not perform well
        #else:
        #    break

    tb_writer.flush()
    tb_writer.close()
    return epoch_loss / len(train_loader)

def seq2seq_generate(model, test_loader, tokenizer, device="cuda:0"):
    generations = []
    
    model = model.to(device)
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        # TODO: get the inputs
        input_ids, target_ids, input_mask = data['input_ids'].to(device), data['output_ids'].to(device), data['input_mask'].to(device)
            
        outputs, _ = model(input_ids, input_mask)

        # TODO: Decode outputs to natural language text
        # Note we expect each output to be a string, not list of tokens here 
        for o_id, output in enumerate(outputs):

            predicted_ids = output.topk(1, dim=1)[1].squeeze(-1).tolist()

            prediction = ' '.join(tokenizer.decode(predicted_ids))

            input_text = ' '.join(tokenizer.decode(input_ids[o_id].tolist()))
            
            reference =  ' '.join(tokenizer.decode(target_ids[o_id].tolist()))

            generations.append({"input": input_text,
                                "reference": reference, 
                                "prediction": prediction})
    
    return generations

def evaluate_rouge(generations):
    # TODO: Implement ROUGE evaluation
    references = [gen['reference'] for gen in generations]
    predictions = [gen['prediction'] for gen in generations] 

    rouge = evaluate.load('rouge')

    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return rouge_scores

def t5_generate(dataset, model, tokenizer, device='cuda:0'):
    # TODO: Implement T5 generation
    generations = []

    for sample in tqdm(dataset, total=len(dataset)):

        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['input_mask'].unsqueeze(0).to(device)
        reference_ids = sample['label_ids']

        max_length = len(input_ids[0]) 
        
        reference = tokenizer.decode(reference_ids, skip_special_tokens=True)

        # Hint: use huggingface text generation
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        prediction = tokenizer.decode(outputs[0],skip_special_tokens=True)

        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        generations.append({
            "input": input_text, 
            "reference": reference, 
            "prediction": prediction})
    
    return generations