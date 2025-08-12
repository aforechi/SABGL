import numpy as np

def MajorityVote(output, rows):
    output = output.reshape((rows, -1))
    predicted = np.zeros(rows, dtype=int)
    for sample in range(rows):
        output_count = np.bincount(output[sample])
        predicted[sample] = np.argmax(output_count)
    return predicted

def MajorityVoteMean(output, expected):
    rows = expected.shape[0]
    predicted = MajorityVote(output, rows)
    return np.mean(np.equal(predicted, expected))

def MajorityVoteAndConfidence(output, rows):
    output = output.reshape((rows, -1))
    confidence =np.zeros(rows, dtype=float)
    predicted = np.zeros(rows, dtype=int)
    for sample in range(rows):
        output_count = np.bincount(output[sample])
        ind1 = np.argmax(output_count)
        max1 = np.max(output_count).astype(float)
        output_count[ind1] = 0
        max2 = np.max(output_count).astype(float)
        predicted[sample] = ind1
        confidence[sample] = (max1 - max2) / max1
    return predicted, confidence

def MajorityVoteClosestToExpected(output, expected):
    rows = expected.shape[0]
    output = output.reshape((rows, -1))
    predicted = np.zeros(rows, dtype=int)
    for sample in range(rows):
        output_count = np.bincount(output[sample])
        top3_most_voted = np.argsort(output_count)[-3:][::-1]
        top3_diff = top3_most_voted - expected[sample]
        predicted[sample] = top3_most_voted[np.argmin(np.absolute(top3_diff))]
    return predicted

def MajorityVoteClosestToPrevious(output, rows):
    output = output.reshape((rows, -1))
    predicted = np.zeros(rows, dtype=int)
    
    output_count = np.bincount(output[0])
    previous = np.argmax(output_count)
    for sample in range(rows):
        output_count = np.bincount(output[sample])
        top3_most_voted = np.argsort(output_count)[-3:][::-1]
        top3_diff = top3_most_voted - previous
        predicted[sample] = top3_most_voted[np.argmin(np.absolute(top3_diff))]
        previous = predicted[sample]
    return predicted

def MajorityVoteClosestToPrevious2(output, rows, groundtruth):
    output = output.reshape((rows, -1))
    predicted = np.zeros(rows, dtype=int)
    
    previous = groundtruth
    for sample in range(rows):
        output_count = np.bincount(output[sample])
        top3_most_voted = np.argsort(output_count)[-3:][::-1]
        top3_diff = top3_most_voted - previous
        predicted[sample] = top3_most_voted[np.argmin(np.absolute(top3_diff))]
        previous = predicted[sample]
    return predicted
