# preprocessing code for comprehensiveness test
# generates the versions of the instances with rationale words removed, and the versions with all non rationale words removed
import copy
import numpy as np
import lime.explanation
import lime.lime_text
import numpy as np
import torch
import gc

BATCH_SIZE = 8

def lime_create_index_arrays(instances, pred_fn, explainer, n_samples=10, k_labels=5):
    """create the indexed strings and index array for the LIME explainer that can then be processed for the faithfulness test
    
    Args:
        instances (list(string)): list of instances to create the indexed strings and index array for
        pred_fn (function): function to get the predictions from the model
        explainer (LimeTextExplainer): the explainer to use to get the explanation
        n_samples (int, optional): number of samples to use for the LIME explainer. Defaults to 10.
        k_labels (int, optional): number of labels to explain. Defaults to 5.
    """
    indexed_strs = np.array([])
    # get the amount of padding needed by finding the longest instance
    # unfourtunately the overall instance length doesn't correspond to the indexed string length, so an additional for loop is needed
    padding_len = 0
    
    # i originally didn't realize padding was an option on the tokenizer so we can probably remove this later
    for instance in instances:
        indexed_str = lime.lime_text.IndexedString(instance)
        inst_len = len(indexed_str.as_np)
        if inst_len > padding_len:
            padding_len = inst_len
    
    # get the single word list version of instances from LIME
    index_array = None
    for i, instance in enumerate(instances):
        indexed_str = lime.lime_text.IndexedString(instance)
        torch.cuda.empty_cache()
        with torch.no_grad():
            exp = explainer.explain_instance(instances[0], pred_fn, num_samples=n_samples, top_labels=k_labels)
        
        # create masked array from map
        exp_map = exp.as_map()

        # get the rationalle words
        for label in exp_map.keys():
            for item in exp_map[label]:
                if index_array is None:
                    index_array = np.array([[i, item[0]]])
                else:
                    # append to the index array so that np.take can be used to mask the data
                    index_array = np.append(index_array, [[i, item[0]]], axis=0)
                    #print(index_array)
        
        # pad and save
        str_as_np = indexed_str.as_np
        
        padding = np.full((padding_len - len(str_as_np)), [''], dtype=str)
        str_as_np = np.append(str_as_np, padding)
        
        if indexed_strs.size == 0:
            indexed_strs = np.array([str_as_np])
        else:
            indexed_strs = np.append(indexed_strs, [str_as_np], axis=0)
        
    index_array_x = np.transpose(index_array)[0]
    index_array_y = np.transpose(index_array)[1]
    index_array = np.array([index_array_x, index_array_y])
    
    return indexed_strs, index_array

def save_indexed_strs(indexed_strs, index_array, file_path):
    """Saves the indexed strings and index array to a npz file

    Args:
        indexed_strs (list[list[string]]): list of instances with words stored as separate strings in a list
        index_array (list[list[int]]): list of indexes storing the explanation for each string
        file_path (string): path to save the npz file
    """
    # save the indexed strings and index array to a json file
    np.savez(file_path, indexed_strs=indexed_strs, index_array=index_array)
    
    
def load_indexed_strs(file_path):
    """Loads the indexed strings and index array from a npz file

    Args:
        file_path (string): path to the npz file to load
    """
    # load the indexed strings and index array from a json file
    with np.load(file_path) as data:
        indexed_strs = data['indexed_strs']
        index_array = data['index_array']
        
    return indexed_strs, index_array

def remove_rationale_words(instances, rationales, join=True, tokenized=False):
    """remove the rationale words from the instances

    Args:
        instances (list(list(string))): list of instances to remove the rationale words from. Each instance is a list of words.
        rationales (list(list(int))): list of rationales to remove from the instances. Each rationale is a list of indexes, where the first index is the instance index and the second index is the word index.
        join (bool, optional): automatically joins the returned string list for the lists with rationale words removed. Defaults to True.

    Returns:
        list(string) (join True) or list(list(string)) (join False) : list of instances with the rationale words removed. Each instance is a list of words or a string.
    """
    inst_rationale_removed = copy.deepcopy(instances)
    
    # TODO: add handling for tokenized data. This will involves masking and editing the inputs key in the dictionary rather than the whole input as it done for 
    # non tokenized inputs
    if tokenized == True:
        rationales_mask = np.zeros(instances['input_ids'].numpy().shape, dtype=bool)
        rationales_mask[rationales[0], rationales[1]] = True
        
        inst_rationale_removed['input_ids'] = torch.from_numpy(np.delete(inst_rationale_removed['input_ids'].numpy(), np.where(rationales_mask), axis=1))
    else:
        rationales_mask = np.zeros(instances.shape, dtype=bool)
        
        # set the values of the rational mask to true based on rationales in a vectorized manner
        # the rationales are in the format [[instance_index_1, instance_index_2, ...], [word_index_1, word_index_2, ...]]
        rationales_mask[rationales[0], rationales[1]] = True
        
        # print(rationales_mask)
        
        # remove the rationale words from the instance in a vectorized manner. The rationale words are a mask, w
        # do this for every instance at the same time using numpy, this is faster than looping through each instance. do not use a list comprehension here
        inst_rationale_removed = np.where(rationales_mask, " ", instances)
        
        if join:
            inst_rationale_removed = [''.join(inst_rationale_removed[i].tolist()) for i in range(len(inst_rationale_removed))]
        
    return inst_rationale_removed

    
def remove_other_words(instances, rationales, join=True, tokenized=False):
    """remove all words that are not in the rationale from the instances

    Args:
        instances (list(list(string))): list of instances to remove the non rationale words from. Each instance is a list of words.
        rationales (list(list(int))): list of rationales to remove from the instances. Each rationale is a list of indexes, where the first index is the instance index and the second index is the word index.
        join (bool, optional): automatically joins the returned string list for the lists with non rationale words removed. Defaults to True.

    Returns:
        list(string) (join True) or list(list(string)) (join False) : list of instances with all non rationale words removed. Each instance is a list of words or a string.
    """
    inst_other_removed = copy.deepcopy(instances)
    
    # TODO: add handling for tokenized data. This will involves masking and editing the inputs key in the dictionary rather than the whole input as it done for 
    # non tokenized inputs
    if tokenized == True:
        inverse_rationales_mask = np.ones(instances['input_ids'].numpy().shape, dtype=bool)
        inverse_rationales_mask[rationales[0], rationales[1]] = False
        
        inst_other_removed['input_ids'] = torch.from_numpy(np.delete(inst_other_removed['input_ids'].numpy(), np.where(rationales_mask), axis=1))
    else:
        # create version of index array where all indexes are added that are not in the rationalle
        inverse_rationales_mask = np.ones(instances.shape, dtype=bool)
        inverse_rationales_mask[rationales[0], rationales[1]] = False
    
        # remove the rationale words from the instance in a vectorized manner
        # do this for every instance at the same time using numpy, this is faster than looping through each instance. do not use a list comprehension here
        # replace each word with "" so that the length of the instance stays the same
        inst_other_removed = np.where(inverse_rationales_mask, " ", instances)
        
        if join:
            inst_other_removed = [''.join(inst_other_removed[i].tolist()) for i in range(len(inst_other_removed))]
        
    return inst_other_removed


def calculate_comprehensiveness(predictions, instances_rationale_removed, model, tokenizer, predictor_func):
    """ Calculate the comprehensiveness of the rationales

    Args:
        predictions (np.array(np.array(float))): List of predictions made with the base instances (no words removed) using the given model.
        instances_rationale_removed (np.array(np.array(string))): List of rationales to compute the comprehensiveness for. This is formatted as a np array of strings.
        model (model): The model to compute the comprehensiveness for.
        tokenizer (tokenizer): The tokenizer to use for the model.
        predictor_func (function): The function to use to get the predictions from the model.
    """
    print("Calculating Comprehensiveness")
    
    # pass the instances through the model - get the predictions
    torch.cuda.empty_cache()
    predictions_rationale_removed = None
    
    for i in range(0, len(instances_rationale_removed), BATCH_SIZE):
        end_range = i + BATCH_SIZE if i + BATCH_SIZE < len(instances_rationale_removed) else len(instances_rationale_removed)
        
        instances_batch = instances_rationale_removed[i:end_range]
        output_batch = predictor_func(instances_batch, model, tokenizer)
        
        if i == 0:
            predictions_rationale_removed = output_batch
        else:
            predictions_rationale_removed = np.concatenate((predictions_rationale_removed, output_batch), axis=0)
            
        gc.collect()
            
    
    # calculate the euclidean distance between the probability of the predicted class and sum over multi labels
    # logits are the classification scores for the opt model
    # confidence_dif = predictions.logits - predictions_rationale_removed.logits
    confidence_dif = predictions - predictions_rationale_removed
    # print("Confidence Dif: ", confidence_dif)
    confidence_dif = np.linalg.norm(confidence_dif, axis=-1)
    # print("Confidence Dif - eudclidean distance: ", confidence_dif)
    
    # return the average confidence difference over the samples
    return np.mean(confidence_dif, axis=-1), confidence_dif


def calculate_sufficency(predictions, instances_other_removed, model, tokenizer, predictor_func):
    """Calculates the sufficiency of the rationales

    Args:
        predictions (np.array(np.array(float))): List of predictions made with the base instances (no words removed) using the given model.
        instances_other_removed (np.array(string)): List of rationales to compute the sufficency for. This is formatted as a list of strings
        model (model): The model to compute the sufficency for.
        tokenizer (tokenizer): The tokenizer to use for the model.
        predictor_func (function): The function to use to get the predictions from the model.
    """
    print("Calculating Sufficiency")
    torch.cuda.empty_cache()
    
    predictions_other_removed = None
    
    for i in range(0, len(instances_other_removed), BATCH_SIZE):
        end_range = i + BATCH_SIZE if i + BATCH_SIZE < len(instances_other_removed) else len(instances_other_removed)
        
        # print("end range: ", end_range)
        # print(i)
        
        instances_batch = instances_other_removed[i:end_range]
        # print(len(instances_batch))

        output_batch = predictor_func(instances_batch, model, tokenizer)
        
        if i == 0:
            predictions_other_removed = output_batch
        else:
            predictions_other_removed = np.concatenate((predictions_other_removed, output_batch), axis=0)
            
        gc.collect()
            
    # predictions_other_removed = predictor_func(instances_other_removed, model, tokenizer)
    # print("Predicitons other removed: ", predictions_other_removed)
    
    # calculate the euclidean distance between the predictions and the predictions with the other words removed
    # logits are the classification scores
    # confidence_dif = predictions.logits - predictions_other_removed.logits
    confidence_dif = predictions - predictions_other_removed
    # print("Confidence Dif: ", confidence_dif)
    confidence_dif = np.linalg.norm(confidence_dif, axis=-1)
    # print("Confidence Dif - eudclidean distance: ", confidence_dif)
    
    # return the average confidence difference
    return np.mean(confidence_dif, axis=-1), confidence_dif
    
    
def calculate_faithfulness(instances, instances_rationalle_removed, instances_other_removed, model, tokenizer, predictor_func):
    """Calculate the faithfulness of the rationales

    Args:
        instances (numpy(numpy(string))): list of instances to compute the faithfulness for. This is formatted as a list of numpy arrays of words.
        instances_rationalle_removed (np.array(string)): list of strings with rationalle words removed. This is formatted as a np array of strings.
        instances_other_removed (np.array(string)): list of instances with all non rationale words removed to compute the faithfulness for. This is formatted as a np array of strings.
        model (model): The model to compute the faithfulness for.
        tokenizer (tokenizer): The tokenizer to use for the model.
        predictor_func (function): The function to use to get the predictions from the model.
    """
    # generate predictions
    redictions = None
    for i in range(0, len(instances), BATCH_SIZE):
        end_range = i + BATCH_SIZE if i + BATCH_SIZE < len(instances) else len(instances)
        
        instances_batch = instances[i:end_range]
        # print(len(instances_batch))
        # print(instances_batch)
        output_batch = predictor_func(instances_batch, model, tokenizer)
        
        if i == 0:
            predictions = output_batch
        else:
            predictions = np.concatenate((predictions, output_batch), axis=0)
            
        gc.collect()
        
    # predictions =  predictor_func(instances, model, tokenizer)
    faithfulness_calc = []
    
    # for each method, calculate the sufficency and comprehensiveness
    for i, instance in enumerate(instances_rationalle_removed):
        print("Currently interpreting instance: ", i)
        
        sufficency, suf_list = calculate_sufficency(predictions, instances_rationalle_removed[i], model, tokenizer, predictor_func)
        comprehensiveness, comp_list = calculate_comprehensiveness(predictions, instances_other_removed[i], model, tokenizer, predictor_func)
        
        # calculate faithfulness
        faithfulness = sufficency * comprehensiveness
        
        print()
        print('-- Metrics -------------------------------------------------------------')
        print()
        
        print()
        print("Faithfulness for iteration: ", faithfulness)
        print("Comprehensiveness for iteration: ", comprehensiveness)
        print("Sufficency for iteration: ", sufficency)
        print()
        
        print()
        print("Comprehensiveness Median: ", np.median(comp_list, axis=-1))
        print("Comprehensiveness q1 (25% percentile): ", np.quantile(comp_list, 0.25, axis=-1))
        print("Comprehensiveness q3 (75% percentile): ", np.quantile(comp_list, 0.75, axis=-1))
        print()
        
        print()
        print("Sufficency Median: ", np.median(suf_list, axis=-1))
        print("Sufficency q1 (25% percentile): ", np.quantile(suf_list, 0.25, axis=-1))
        print("Sufficency q3 (75% percentile): ", np.quantile(suf_list, 0.75, axis=-1))
        print()
        
        faithfulness_calc.append(faithfulness)
    
    # return the minimum index of the faithfulness_calc to get the best method
    return np.argmin(faithfulness_calc), faithfulness_calc
        
    
    
