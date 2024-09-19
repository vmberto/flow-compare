import pandas as pd
import re

with open('log_likelihood_results.txt', 'r') as file:
    raw_data = file.read()

pattern = re.compile(r'^Fold (\d+): Average Log-Likelihood on (.+?)_(\d+): (-?\d+\.\d+)$')

data = []
for line in raw_data.strip().split('\n'):
    match = pattern.match(line.strip())
    if match:
        fold = int(match.group(1))
        corruption = match.group(2)
        severity = int(match.group(3))
        log_likelihood = float(match.group(4))
        data.append({
            'Fold': fold,
            'Corruption': corruption,
            'Severity': severity,
            'LogLikelihood': log_likelihood
        })
        print(data)

df = pd.DataFrame(data)
df.to_csv('log_likelihood.csv')