import numpy as np
import matplotlib.pyplot as plt

#coin toss probability
coinCount = np.random.choice(['H','T'], size = 1000)

heads = np.where(coinCount == 'H', 1, 0)

runningProb = np.cumsum(heads) / np.arange(1, 1001)

values, counts = np.unique(coinCount, return_counts=True )


#dice count

diceCount = np.random.randint(1,7, size=10000)

dicevalues, dicecounts = np.unique(diceCount, return_counts=True)

plt.figure(figsize=(12, 4))


plt.subplot(1, 2, 1)
coinBars = plt.bar(values, counts, color=['steelblue', 'coral'])
plt.bar_label(coinBars, padding=3)
plt.axhline(y=500, color='red', linestyle='--', label='Expected 500')
plt.title("CoinToss H vs T Frequency")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.legend()


plt.subplot(1, 2, 2)
diceBars = plt.bar(dicevalues, dicecounts, color=['steelblue', 'coral'])
plt.bar_label(diceBars, padding=3)
plt.title("Dice")
plt.xlabel("Outcome")
plt.ylabel("Count")


plt.tight_layout()
plt.show()