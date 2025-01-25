import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import time
import iou_calc
import time
import shutil
import os

def custom_collate_fn_fixed(batch, context_length= 10):
    # Split the batch into inputs, sequence_names, and targets
    inputs, targets, sequence_names = zip(*batch)
    
    # Truncate or pad inputs to the context length (10 in this case)
    truncated_inputs = []
    for seq in inputs:
        if len(seq) > context_length:
            truncated_inputs.append(seq[:context_length])  # Truncate to context length
        else:
            padding = torch.zeros(context_length - len(seq), *seq.shape[1:])  # Create padding tensor of appropriate shape
            truncated_inputs.append(torch.cat([seq, padding], dim=0))  # Pad with zeros
    
    # Stack padded inputs and targets
    inputs_padded = torch.stack(truncated_inputs, dim=0)  # Shape: (batch_size, context_length, feature_size)
    targets = torch.stack(targets)  # Assuming targets are tensors
    
    # Create input lengths to represent actual sequence lengths before padding
    input_lengths = torch.tensor([min(len(seq), context_length) for seq in inputs], dtype=torch.long)
    
    # Create a mask to indicate valid data vs padded data
    mask = (inputs_padded != 0)
    
    return inputs_padded, targets, sequence_names, input_lengths, mask




# Custom collate function for handling batches with variable-length inputs
def custom_collate_fn_var(batch):
    # Split the batch into inputs, sequence_names, and targets
    inputs, targets, sequence_names = zip(*batch)
    # print("input shape is : ", inputs.shape)
    # Pad the inputs (they are assumed to be lists of tensors or tensors themselves)
    # Find the maximum length of the inputs
    inputs_padded = pad_sequence(inputs, batch_first=True)
    # print(" padded inputs", padded_inputs.shape)
    # Convert the sequence_names and targets back to tensors if they are lists
    # sequence_names = torch.stack(sequence_names)  # Assuming sequence_names are tensors
    targets = torch.stack(targets)  # Assuming targets are tensors
    
    ## input lengths is th emask 
    input_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long) 
    mask = (inputs_padded != 0)  # Mask to ignore padded zeros during training


    return inputs_padded, targets, sequence_names, input_lengths, mask






def train_var_window(args, model, train_dataloader, val_dataloader, configs, output_dir):
    print("hellow?")

    # best_model_path = "best_model_bbox_{}_{}.pth".format(args.dataset, args.model_type)
    best_loss = float('inf')
    max_window = configs["max_window"]
    num_blocks = configs["num_blocks"]
    current_time = time.localtime()
    formatted_date = time.strftime("%d %B", current_time)
    formatted_date = str(formatted_date.split(' ')[0])+ "_" + str(formatted_date.split(' ')[1])
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(os.getcwd(), output_dir))
    best_model_name  = "{}/best_model_bbox_{}_{}_{}_max_window_{}_num_blocks_{}_{}".format(output_dir, args.dataset, args.window_size, args.model_type, max_window, num_blocks, formatted_date) 
    # best_model_path = "best_model_bbox_mot20_{}.pth".format(configs['model_used'])
    best_model_path =  best_model_name + ".pth"
    best_loss = float('inf')
    
    device = configs['device']    
    (criterion, criterion_2) = configs['criterion']
    # criterion_2 = iou_calc.CIOU_Loss_Perplexity
    optimizer = configs['optimizer']
    scheduler = configs['scheduler']
    num_epochs = configs['epochs']
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0  # Initialize epoch_loss
        model.train()
        epoch_loss_criterion_1 = 0.0
        epoch_loss_criterion_2 =0.0
        for idx, (inputs, targets, sequences, input_lengths, masks) in enumerate(train_dataloader):
            if idx ==0:
                print(" shape pf inputs is : ", inputs.shape)
            # sorted_lengths, perm_idx = input_lengths.sort(0, descending=True)
            # print(" perm idx", perm_idx)
            # inputs_padded = inputs[perm_idx]
            # # target = targets[perm_idx]
            # print("sorted lengths : ", sorted_lengths)
            # print("input padded : ", inputs.shape)
            # masks = masks.to(device)
            # print(" masks is : ", masks.shape)
            # packed_input = pack_padded_sequence(inputs_padded, sorted_lengths, batch_first=True)
            # print(" packed input type is :", type(packed_input))
            # Move tensors to the configured device
            # print("inputs shape is :", inputs)
            # packed_input = packed_input.to(device)
            # target = target.to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            # print("shape of inputs is : ", inputs.shape)
            targets = targets.float()
            # print(" targets are : ", targets)
            # Forward pass
            # continue
            outputs = model(inputs.float())
            # outputs = outputs.to(device)
            
            # print(" outputs shape is :", outputs.shape)
            # exit(0)
            # outputs = pad_packed_sequence(outputs, batch_first = True)
            # print(" outputs are : ", outputs)
            # print(" shape of outputs is : ", outputs.shape)
            # print(" shape of targets is : ", targets.shape)
            # continue
            loss_criterion_1 = criterion(outputs, targets)
            # print(" MSE Loss shape is : ", loss_mse.shape)
            # loss_mse = loss_mse * masks ## Broadcast the mask over the loss Tensor
            
            loss_criterion_2 = criterion_2(outputs, targets)
            # loss_giou_func = loss_giou_func * masks
            
            # continue
            
            total_loss = configs['lambda_criterion_1'] * loss_criterion_1+ configs['lambda_criterion_2'] * loss_criterion_2
            # total_loss  = total_loss.sum()/masks.sum()
            
            # combined_loss = giou_weight * giou + mse_weight * mse
            
            epoch_loss_criterion_2 +=loss_criterion_2
            epoch_loss_criterion_1 +=loss_criterion_1
            epoch_loss += total_loss# Accumulate loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            # loss_smooth_l1.backward()
            loss_criterion_2.backward(retain_graph = True)
            loss_criterion_1.backward()
            if torch.isnan(total_loss):
                # print("")
                print(" targets are :", targets)
                print("predictions are : ", prediction_offset)
                print("MSE Loss for this prediction is : ", loss_criterion_1.item())
                print("CIOU loss is : ", loss_criterion_2)
                exit(0)
            # total_loss.backward()
            
            optimizer.step()
            scheduler.step()
            # if scheduler.current_step %4000 == 0:
            #     print("Current step is :  {} and the learning rate is : {}".format(scheduler.current_step, optimizer.param_groups[-1]['lr']))
                # print(" input is : ", inputs)

            # Step the warmup scheduler
            # if warmup_scheduler.current_step < warmup_scheduler.warmup_steps:
            #     warmup_scheduler.step()
            # else:
            #     # Step the standard scheduler after warmup
            #     scheduler_after_warmup.step()
            
            # Update the learning rate
            
        print(" {} : {}".format(criterion_2.__name__, epoch_loss_criterion_2/len(train_dataloader)))
        print(" {}: {}".format(criterion.__class__.__name__, epoch_loss_criterion_1/len(train_dataloader)))
        
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():  ## No gradient so we dont update the weights and biases with test
            for data_valid, targets_valid, sequences, input_lengths, masks in val_dataloader:
                data_valid, targets_valid  = data_valid.to(device), targets_valid.to(device)
                
                targets_valid = targets_valid.float()
                
                prediction_offset = model(data_valid.float()).to(device)
                
                loss_criterion_1 = criterion(prediction_offset, targets_valid)
                loss_criterion_2= criterion_2(prediction_offset, targets_valid)
            
                total_loss = loss_criterion_1 + loss_criterion_2
                if torch.isnan(total_loss):
                    # print("")
                    print(" targets are :", targets_valid)
                    print("predictions are : ", prediction_offset)
                    print("MSE Loss for this prediction is : ", loss_criterion_1.item())
                    print("CIOU loss is : ", loss_criterion_2)
                    exit(0)
                validation_loss += total_loss
                
        print("Accumulated training loss is : ", epoch_loss)
        avg_loss = epoch_loss / len(train_dataloader)  # Calculate average loss for the epoch
        avg_valid_loss = validation_loss / len(val_dataloader)
        
        if abs(avg_valid_loss) < abs(best_loss):
            best_loss = avg_valid_loss
            if args.save_model:
                # best_model_file = best_model_path.split(".")[0] +   ".pth"
                # best_model_file = best_model_path.split(".")[0] + "_" + str(epoch) + ".pth"

                torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with loss: {best_loss:.4f} with name : {best_model_path}')
        end_time = time.time()
        time_taken = end_time - start_time
        if args.run_wandb:
            wandb.log({'epoch': epoch + 1, 'training loss': avg_loss, 'validation loss': avg_valid_loss})

        print('Epoch [{}/{}], Train Loss: {} , Validation Loss : {} , Best Loss : {}, Time Taken : {}'.format(epoch+1, num_epochs, avg_loss, avg_valid_loss, best_loss, time_taken))
    # shutil.move(best_model_path, best_model_name+ "_" + formatted_date + ".pth")



def train_const_window(args, model, train_dataloader, val_dataloader, configs):
    # print("hellow?")
    print("you have selected CONSTANT window size of {}".format(configs['window_size']))
    
    best_model_path = "best_model_bbox_{}_{}_{}.pth".format(args.dataset, args.window_size,  args.model_type)
    best_loss = float('inf')
    device = configs['device']    
    
    criterion = configs['criterion']
    optimizer = configs['optimizer']
    scheduler = configs['scheduler']

    num_epochs = configs['epochs']
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0  # Initialize epoch_loss
        model.train()
        epoch_loss_mse = 0.0
        epoch_loss_giou =0.0
        for inputs, targets, sequences in train_dataloader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            # print("shape of inputs is : ", inputs.shape)
            target = targets.float()
            # print(" targets are : ", targets)
            # Forward pass
            outputs = model(inputs.float())
            # outputs = outputs.to(device)
            
            # print(" outputs shape is :", outputs.shape)
            # exit(0)
            # outputs = pad_packed_sequence(outputs, batch_first = True)
            # print(" outputs are : ", outputs)
            # print(" shape of outputs is : ", outputs.shape)
            # print(" shape of targets is : ", targets.shape)
            # continue
            loss_mse = criterion(outputs, target)
            # print(" MSE Loss shape is : ", loss_mse.shape)
            
            loss_giou_func = iou_calc.ciou_loss(outputs, targets)
            
            
            total_loss = configs['lambda_giou'] * loss_giou_func + configs['lambda_mse'] * loss_mse
            # total_loss  = total_loss.sum()/masks.sum()
            
            # combined_loss = giou_weight * giou + mse_weight * mse
            
            epoch_loss_giou +=loss_giou_func    
            epoch_loss_mse +=loss_mse
            epoch_loss += total_loss# Accumulate loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            # loss_smooth_l1.backward()
            loss_giou_func.backward(retain_graph = True)
            loss_mse.backward()
            
            # total_loss.backward()
            
            optimizer.step()
            scheduler.step()
            # Step the warmup scheduler
            # if warmup_scheduler.current_step < warmup_scheduler.warmup_steps:
            #     warmup_scheduler.step()
            # else:
            #     # Step the standard scheduler after warmup
            #     scheduler_after_warmup.step()
            
            # Update the learning rate
            
        print(" MSE Loss : ", epoch_loss_mse/len(train_dataloader))
        print(" GIOU Loss: ", epoch_loss_giou/len(train_dataloader))
        
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():  ## No gradient so we dont update the weights and biases with test
            for data_valid, targets_valid, sequences in val_dataloader:
                data_valid, targets_valid  = data_valid.to(device), targets_valid.to(device)
                
                targets_valid = targets_valid.float()
                
                prediction_offset = model(data_valid.float()).to(device)
                
                loss_mse = criterion(prediction_offset, targets_valid)
                loss_giou = iou_calc.ciou_loss(prediction_offset, targets_valid)
            
                total_loss = loss_mse + loss_giou
                validation_loss += total_loss
                
        print("Accumulated training loss is : ", epoch_loss)
        avg_loss = epoch_loss / len(train_dataloader)  # Calculate average loss for the epoch
        avg_valid_loss = validation_loss / len(val_dataloader)
        
        if abs(avg_valid_loss) < abs(best_loss):
            best_loss = avg_valid_loss
            if args.save_model:
                # best_model_file = best_model_path.split(".")[0] + "_" + str()
                # best_model_file = best_model_path.split(".")[0] + "_" + str(epoch) + ".pth"
                torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with loss: {best_loss:.4f}')
        end_time = time.time()
        time_taken = end_time - start_time
        if args.run_wandb:
            wandb.log({'epoch': epoch + 1, 'training loss': avg_loss, 'validation loss': avg_valid_loss})

        print('Epoch [{}/{}], Train Loss: {} , Validation Loss : {} , Best Loss : {}, Time Taken : {}'.format(epoch+1, num_epochs, avg_loss, avg_valid_loss, best_loss, time_taken))




