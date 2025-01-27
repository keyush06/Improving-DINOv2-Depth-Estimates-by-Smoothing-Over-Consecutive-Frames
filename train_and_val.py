from utils import *

def forward_pass(selected_data, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, batch_size, model_string):

    ground_truth_depth = torch.zeros(batch_size, image_height, image_width)
    transformed_image_current = torch.zeros(batch_size, 3, image_height, image_width)
    grayscale_image_current = np.zeros((batch_size, image_height, image_width))

    for idx in range(batch_size):
        selected_dataset, selected_image = selected_data[idx]
        ground_truth_depth[idx] = 10.0 * torch.tensor(np.array(Image.open(data_directory + '/' + selected_dataset + '/' + str(selected_image+1) + '.png')).astype(float)/255.0, dtype=torch.float32)
        transformed_image_current[idx] = transform(Image.open(data_directory + '/' + selected_dataset + '/' + str(selected_image+1) + '.jpg'))
        grayscale_image_current[idx] = cv2.imread(data_directory + '/' + selected_dataset + '/' + str(selected_image+1) + '.jpg', 0)

    with torch.no_grad():
        original_dino_depth_map = model.whole_inference(transformed_image_current.cuda(), img_meta=None, rescale=True).squeeze()

        dino_features_current = model.extract_feat(transformed_image_current.cuda())

    original_mse = F.mse_loss(original_dino_depth_map.cpu(), ground_truth_depth).item()

    transformed_image_previous = torch.zeros(batch_size, 3, image_height, image_width)
    for idx in range(batch_size):
        selected_dataset, selected_image = selected_data[idx]
        transformed_image_previous[idx] = transform(Image.open(data_directory + '/' + selected_dataset + '/' + str(selected_image) + '.jpg'))

    with torch.no_grad():
        dino_features_previous = model.extract_feat(transformed_image_previous.cuda())

    shift_tensor_conv_x = torch.zeros(batch_size, 1, feature_height, feature_width).cuda()
    shift_tensor_conv_y = torch.zeros(batch_size, 1, feature_height, feature_width).cuda()
    shift_tensor_fc_x = torch.zeros(batch_size, 1).cuda()
    shift_tensor_fc_y = torch.zeros(batch_size, 1).cuda()

    for idx in range(batch_size):
        selected_dataset, selected_image = selected_data[idx]

        grayscale_image_previous = cv2.imread(data_directory + '/' + selected_dataset + '/' + str(selected_image) + '.jpg', 0)
        shift_x, shift_y = cv2.phaseCorrelate(np.float32(grayscale_image_previous), np.float32(grayscale_image_current[idx]))[0]

        shift_tensor_conv_x[idx] = shift_x * torch.ones(1, 1, feature_height, feature_width)
        shift_tensor_conv_y[idx] = shift_y * torch.ones(1, 1, feature_height, feature_width)
        shift_tensor_fc_x[idx] = shift_x * torch.ones(1, 1)
        shift_tensor_fc_y[idx] = shift_y * torch.ones(1, 1)

    dino_features_new = []
    for dpt_layer in range(4):
        adapter_input_image = torch.cat((dino_features_current[dpt_layer][0].detach().clone(), dino_features_previous[dpt_layer][0].detach().clone(), shift_tensor_conv_x, shift_tensor_conv_y), dim=1)
        adapter_input_fc = torch.cat((dino_features_current[dpt_layer][1].detach().clone(), dino_features_previous[dpt_layer][1].detach().clone(), shift_tensor_fc_x, shift_tensor_fc_y), dim=1)

        dino_features_new_x = []
        if 'cnn' in model_string:
            dino_features_new_x.append(adapter_model.forward_conv(adapter_input_image, dpt_layer))
        else:
            dino_features_new_x.append(adapter_model.forward_transformer(adapter_input_image, dpt_layer))
        dino_features_new_x.append(adapter_model.forward_fc(adapter_input_fc, dpt_layer))
        dino_features_new.append(dino_features_new_x)

    updated_dino_depth_map = model._decode_head_forward_test(dino_features_new, img_metas=None)

    updated_dino_depth_map = torch.clamp(updated_dino_depth_map, min=model.decode_head.min_depth, max=model.decode_head.max_depth)

    updated_dino_depth_map = F.interpolate(updated_dino_depth_map, transformed_image_current.shape[2:], None, "bilinear", model.align_corners).squeeze()

    loss = F.mse_loss(updated_dino_depth_map.cpu(), ground_truth_depth)

    updated_mse = loss.item()

    if 'cnn_regularized' in model_string:
        lam = 0.1
        loss += lam * F.mse_loss(updated_dino_depth_map.cpu(), original_dino_depth_map.cpu())

    del ground_truth_depth, transformed_image_current, dino_features_current, original_dino_depth_map, grayscale_image_current
    del adapter_input_image, adapter_input_fc, dino_features_new, dino_features_new_x, updated_dino_depth_map
    del shift_tensor_conv_x, shift_tensor_conv_y, shift_tensor_fc_x, shift_tensor_fc_y, shift_x, shift_y, grayscale_image_previous
    del transformed_image_previous, dino_features_previous
    torch.cuda.empty_cache()

    return original_mse, updated_mse, loss


def val_loop(val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, num_images_val, batch_size, epoch, model_string, end_of_epoch=False, training_percentage=None):
    num_batches_val = int(num_images_val/batch_size)
    original_mse_current_epoch_val_list = torch.zeros(num_batches_val)
    updated_mse_current_epoch_val_list = torch.zeros(num_batches_val)

    if end_of_epoch:
        index_string = 'End of Epoch ' + str(epoch)
    else:
        index_string = 'Epoch ' + str(epoch) + ', ' + str(training_percentage) + '% Training Completed'

    random.shuffle(val_list)
    for i in range(num_batches_val):
        selected_data = val_list[i*batch_size:i*batch_size+batch_size]

        original_mse, updated_mse, _ = forward_pass(selected_data, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, batch_size, model_string)

        original_mse_current_epoch_val_list[i] = original_mse

        updated_mse_current_epoch_val_list[i] = updated_mse

        print("Validation - " + index_string + ", Images " + str(i*batch_size) + "-" + str(i*batch_size+batch_size-1) + ": Original MSE = " + str(original_mse) + ", Updated MSE = " + str(updated_mse))

        del original_mse, updated_mse
        torch.cuda.empty_cache()

    if num_images_val % batch_size == 0:
        original_mse_current_epoch_val_mean = original_mse_current_epoch_val_list.mean().item()
        updated_mse_current_epoch_val_mean = updated_mse_current_epoch_val_list.mean().item()
        difference_current_epoch_val_mean = updated_mse_current_epoch_val_mean - original_mse_current_epoch_val_mean

    else:
        selected_data = val_list[num_batches_val*batch_size:]

        original_mse, updated_mse, _ = forward_pass(selected_data, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, num_images_val % batch_size, model_string)

        print("Validation - " + index_string + ", Images " + str(num_batches_val*batch_size) + "-" + str(num_images_val-1) + ": Original MSE = " + str(original_mse) + ", Updated MSE = " + str(updated_mse))

        original_mse_current_epoch_val_mean = (original_mse_current_epoch_val_list.mean().item() * num_batches_val * batch_size + original_mse * (num_images_val % batch_size)) / num_images_val
        updated_mse_current_epoch_val_mean = (updated_mse_current_epoch_val_list.mean().item() * num_batches_val * batch_size + updated_mse * (num_images_val % batch_size)) / num_images_val
        difference_current_epoch_val_mean = updated_mse_current_epoch_val_mean - original_mse_current_epoch_val_mean

    print("Validation - " + index_string + ": Original MSE at Epoch " + str(epoch) + " = " + str(original_mse_current_epoch_val_mean))
    print("Validation - " + index_string + ": Updated MSE at Epoch " + str(epoch) + " = " + str(updated_mse_current_epoch_val_mean))
    print("Validation - " + index_string + ": Difference at Epoch " + str(epoch) + " = " + str(difference_current_epoch_val_mean))

    return original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean


def train_loop(original_mse_val, updated_mse_val, train_list, val_list, data_directory, transform, model, adapter_model, optimizer, image_height, image_width, feature_height, feature_width, num_images_train, num_images_val, batch_size, batches_per_backprop, epoch, val_checkpoints, val_flag, model_string, start_batch_index=0):
    num_batches_train = int(num_images_train/batch_size)

    original_train_file_path_current_epoch = './training_logs/original_mse_current_epoch_train_' + model_string + '.pt'
    updated_train_file_path_current_epoch = './training_logs/updated_mse_current_epoch_train_' + model_string + '.pt'
    train_list_file_path = './train_val_split/train_list_current_epoch_' + model_string + '.json'

    if start_batch_index == 0:
        original_mse_current_epoch_train_list = torch.zeros(num_batches_train)
        updated_mse_current_epoch_train_list = torch.zeros(num_batches_train)
        random.shuffle(train_list)
    else:
        original_mse_current_epoch_train_list = torch.load(original_train_file_path_current_epoch, weights_only=True)
        updated_mse_current_epoch_train_list = torch.load(updated_train_file_path_current_epoch, weights_only=True)
        with open(train_list_file_path, "r") as file:
            train_list = json.load(file)

    original_running_train_loss = 0.0
    updated_running_train_loss = 0.0
    for i in range(start_batch_index, num_batches_train):
        selected_data = train_list[i*batch_size:i*batch_size+batch_size]

        original_mse, updated_mse, loss = forward_pass(selected_data, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, batch_size, model_string)

        original_running_train_loss += original_mse
        original_mse_current_epoch_train_list[i] = original_mse

        updated_running_train_loss += updated_mse
        updated_mse_current_epoch_train_list[i] = updated_mse

        loss = loss / batches_per_backprop
        loss.backward()
        if i % batches_per_backprop == batches_per_backprop - 1:
            print("Training Epoch " + str(epoch) + ", Images " + str(i*batch_size) + "-" + str(i*batch_size+batch_size-1) + ": Original MSE = " + str(original_running_train_loss/batches_per_backprop) + ", Updated MSE = " + str(updated_running_train_loss/batches_per_backprop))
            optimizer.step()
            optimizer.zero_grad()
            original_running_train_loss = 0.0
            updated_running_train_loss = 0.0

        del original_mse, updated_mse
        torch.cuda.empty_cache()

        if val_flag < len(val_checkpoints):
            if i == val_checkpoints[val_flag]:
                torch.save(adapter_model.state_dict(), './models/' + model_string + '_model_epoch_' + str(epoch) + '_segment_' + str(val_flag + 1) + '.pth')

                torch.save(original_mse_current_epoch_train_list, original_train_file_path_current_epoch)
                torch.save(updated_mse_current_epoch_train_list, updated_train_file_path_current_epoch)

                with open(train_list_file_path, "w") as file:
                    json.dump(train_list, file)

                original_mse_current_epoch_val_mean, updated_mse_current_epoch_val_mean = val_loop(val_list, data_directory, transform, model, adapter_model, image_height, image_width, feature_height, feature_width, num_images_val, batch_size, epoch, model_string, end_of_epoch=False, training_percentage=round(100*val_checkpoints[val_flag]/num_batches_train))
                original_mse_val[epoch, val_flag] = original_mse_current_epoch_val_mean
                updated_mse_val[epoch, val_flag] = updated_mse_current_epoch_val_mean
                val_flag += 1

                torch.save(original_mse_val, './training_logs/original_mse_val_' + model_string + '.pt')
                torch.save(updated_mse_val, './training_logs/updated_mse_val_' + model_string + '.pt')

    original_mse_current_epoch_train_mean = original_mse_current_epoch_train_list.mean().item()
    updated_mse_current_epoch_train_mean = updated_mse_current_epoch_train_list.mean().item()
    difference_current_epoch_train_mean = updated_mse_current_epoch_train_mean - original_mse_current_epoch_train_mean

    print("Training: Original MSE at Epoch " + str(epoch) + " = " + str(original_mse_current_epoch_train_mean))
    print("Training: Updated MSE at Epoch " + str(epoch) + " = " + str(updated_mse_current_epoch_train_mean))
    print("Training: Difference at Epoch " + str(epoch) + " = " + str(difference_current_epoch_train_mean))

    return original_mse_current_epoch_train_mean, updated_mse_current_epoch_train_mean, original_mse_val, updated_mse_val
