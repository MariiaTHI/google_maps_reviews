from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ausland_reviews.csv')
all_text = ' '.join(df['text_review'].astype(str))
# sample_text = ' '.join(df['text_column'].sample(n=1000).astype(str))  # Adjust 'n' as needed


wordcloud = WordCloud(background_color="white").generate(all_text)

# Display the generated image
plt.figure(figsize=(10, 10), facecolor=None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
