import numpy as np

# prototypes (CDOST order)
prototypes = {
    'Choleric'   : np.array([3.,5.,1.,2.,5.]),
    'Melancholic': np.array([5.,1.,3.,4.,1.]),
    'Phlegmatic' : np.array([1.,2.,5.,5.,1.]),
    'Sanguine'   : np.array([1.,5.,3.,3.,3.])
}

def pop_variance(diff):
    mu = diff.mean()            # population mean of the diff vector
    return ((diff - mu)**2).mean()  # population variance (divide by N)

def classify_subject(subject_vec, prototypes, blend_threshold=0.5):
    results = {}
    for name, proto in prototypes.items():
        diff = subject_vec - proto
        results[name] = pop_variance(diff)
    # sort ascending (lower variance => better fit)
    sorted_results = sorted(results.items(), key=lambda kv: kv[1])
    primary, var_primary = sorted_results[0]
    secondary, var_secondary = sorted_results[1]
    delta = var_secondary - var_primary
    if delta < blend_threshold:
        label = f"Blend({primary}-{secondary})"
        strength = "strong" if delta < (blend_threshold/2) else "weak"
    else:
        label = primary
        strength = "dominant"
    return {'label': label, 'strength': strength,
            'primary': (primary,var_primary),
            'secondary': (secondary,var_secondary),
            'all': results}

# Example subject (CDOST)
subject = np.array([3.,2.,2.,2.,2.])  # the example in your doc
out = classify_subject(subject, prototypes, blend_threshold=0.5)
print(out)
