# preprocessing code for comprehensiveness test
# generates the versions of the instances with rationale words removed, and the versions with all non rationale words removed
import copy
import numpy as np
import lime.explanation
import lime.lime_text
import numpy as np
import torch



def lime_create_index_arrays(instances, pred_fn, explainer, n_samples=10, k_labels=5):
    """get the explanation for the given instances and generate index arrays for the rationale"""
    indexed_strs = np.array([])
    # get the amount of padding needed by finding the longest instance
    # unfourtunately the overall instance length doesn't correspond to the indexed string length, so an additional for loop is needed
    padding_len = 0
    
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


def remove_rationale_words(instances, rationales, join=True):
    inst_rationale_removed = copy.deepcopy(instances)
    
    rationales_mask = np.zeros(instances.shape, dtype=bool)
    
    # set the values of the rational mask to true based on rationales in a vectorized manner
    # the rationales are in the format [[instance_index_1, instance_index_2, ...], [word_index_1, word_index_2, ...]]
    rationales_mask[rationales[0], rationales[1]] = True
    
    print(rationales_mask)
    
    # remove the rationale words from the instance in a vectorized manner. The rationale words are a mask, w
    # do this for every instance at the same time using numpy, this is faster than looping through each instance. do not use a list comprehension here
    inst_rationale_removed = np.where(rationales_mask, " ", instances)
    
    if join:
        inst_rationale_removed = [''.join(inst_rationale_removed[i].tolist()) for i in range(len(inst_rationale_removed))]
        
    return inst_rationale_removed

    
def remove_other_words(instances, rationales, join=True):
    inst_other_removed = copy.deepcopy(instances)
    
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
        instances_rationale_removed (np.array(np.array(word))): List of rationales to compute the comprehensiveness for. This is formatted as a list of numpy arrays, where each array is an array of words.
        model (model): The model to compute the comprehensiveness for.
    """
    print("Calculating Comprehensiveness")
    
    # pass the instances through the model - get the predictions
    torch.cuda.empty_cache()
    predictions_rationale_removed =  predictor_func(instances_rationale_removed, model, tokenizer)
    print("Predictions ratonale removed: ", predictions_rationale_removed)
    
    # calculate the euclidean distance between the probability of the predicted class and sum over multi labels
    # logits are the classification scores for the opt model
    # confidence_dif = predictions.logits - predictions_rationale_removed.logits
    confidence_dif = predictions - predictions_rationale_removed
    print("Confidence Dif: ", confidence_dif)
    confidence_dif = np.linalg.norm(confidence_dif, axis=-1)
    print("Confidence Dif - eudclidean distance: ", confidence_dif)
    
    # return the average confidence difference over the samples
    return np.mean(confidence_dif, axis=-1)


def calculate_sufficency(predictions, instances_other_removed, model, tokenizer, predictor_func):
    """Calculates the sufficiency of the rationales

    Args:
        predictions (np.array(np.array(float))): List of predictions made with the base instances (no words removed) using the given model.
        instances_rationale_removed (np.array(np.array(indices))): List of rationales to compute the sufficency for. This is formatted as a list of numpy arrays, where each array acts as a mask, where a 1 indicates that the word is a rationale word.
        model (model): The model to compute the sufficency for.
    """
    print("Calculating Sufficiency")
    torch.cuda.empty_cache()
    predictions_other_removed = predictor_func(instances_other_removed, model, tokenizer)
    print("Predicitons other removed: ", predictions_other_removed)
    
    # calculate the euclidean distance between the predictions and the predictions with the other words removed
    # logits are the classification scores
    # confidence_dif = predictions.logits - predictions_other_removed.logits
    confidence_dif = predictions - predictions_other_removed
    print("Confidence Dif: ", confidence_dif)
    confidence_dif = np.linalg.norm(confidence_dif, axis=-1)
    print("Confidence Dif - eudclidean distance: ", confidence_dif)
    
    # return the average confidence difference
    return np.mean(confidence_dif, axis=-1)
    
    
def calculate_faithfulness(instances, instances_rationalle_removed, instances_other_removed, model, tokenizer, predictor_func):
    """Calculate the faithfulness of the rationales

    Args:
        instances (numpy(numpy(string))): List of instances to compute the faithfulness for. This is formatted as a list of numpy arrays of words.
        instances_rationalle_removed (numpy(numpy(numpy(int)))): List of rationales to compute the faithfulness for. This is formatted as a list of numpy arrays, where each array acts as a mask, where a 1 indicates that the word is a rationale word. Each list is provided by one interpretability method.
        instances_other_removed (numpy(numpy(int))): List of instances with all non rationale words removed to compute the faithfulness for. This is formatted as a list of numpy arrays, where each array acts as a mask, where a 1 indicates that the word is not a rationale word. Each list is provided by one interpretability method.
        model (model): The model to compute the faithfulness for.
    """
    # generate predictions
    predictions =  predictor_func(instances, model, tokenizer)
    faithfulness_calc = []
    
    # for each method, calculate the sufficency and comprehensiveness
    for i, instance in enumerate(instances_rationalle_removed):
        print("Currently interpreting instance: ", i)
        
        print(instances_rationalle_removed[i])
        sufficency = calculate_sufficency(predictions, instances_rationalle_removed[i], model, tokenizer, predictor_func)
        print("Sufficency for iteration: ", sufficency)
        
        comprehensiveness = calculate_comprehensiveness(predictions, instances_other_removed[i], model, tokenizer, predictor_func)
        print("Comprehensiveness for iteration: ", comprehensiveness)
        
        # calculate faithfulness
        faithfulness = sufficency * comprehensiveness
        print("Faithfulness for iteration: ", faithfulness)
        faithfulness_calc.append(faithfulness)
    
    # return the minimum index of the faithfulness_calc to get the best method
    return np.argmin(faithfulness_calc), faithfulness_calc
        
    
    
