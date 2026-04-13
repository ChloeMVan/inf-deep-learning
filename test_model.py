import torch

def test_model(model, data, device='cpu', do_print=False):
    model.eval()
    total = 0
    correct = 0
    labels = ["Real", "Fake"]
    record = []
    for i, (frames, label) in enumerate(data):
      frames = frames.unsqueeze(0).to(device) # add batch dimension
      output = model(frames)[0] # since batch size is 1, we will get only one output
      y_pred = torch.round(output).item()
      y_actual = label
      record.append((y_pred, y_actual))
      if y_actual == y_pred:
        correct = correct+1
      total = total+1
    if do_print:  
        print(f"Accuracy = {correct / total * 100:.4f} %")
        for i in range(2):
            tps = sum([1 for x in record if x[0] == i and x[1] == i])
            fps = sum([1 for x in record if x[0] == i and x[1] != i])
            fns = sum([1 for x in record if x[0] != i and x[1] == i])
            tns = sum([1 for x in record if x[0] != i and x[1] != i])
            guesses = [x[0] for x in record if x[1] == i]
            # represents what the model guessed when this was the actual label
            counts = [guesses.count(j) for j in range(2)]
            if tps == 0:
                precision = 0.0
                recall = 0.0
                f_score = 0.0
            else:
                precision = tps / (tps + fps)
                recall = tps / (tps + fns)
                f_score = 2 * precision * recall / (precision + recall)
            print(f"  {labels[i]} ({i}):\n    F Score: {f_score:.4f}\n    Guesses: {counts}\n    Precision: {precision:.4f}\n    Recall: {recall:.4f}")
            if i == 1:
                print(f"\tPred Pos.\tPred Neg.\n\tReal Pos.\t{tps}\t{fns}\n\tReal Neg.\t{fps}\t{tns}")
            
    else:
        return correct / total * 100    
