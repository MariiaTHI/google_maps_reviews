import googlemaps
import pandas as pd
from datetime import datetime

#init Google maps Client 
key = 'MY_GOOGLE_API_KEY' # Replace with your API key 
gmaps = googlemaps.Client(key=key)  

def get_place_details(place_id, api_key):
    gmaps = googlemaps.Client(key=api_key)
    place_details = gmaps.place(place_id=place_id, fields=['name'])
    return place_details.get('result', {}).get('name', 'Unknown')

def fetch_reviews(place_id):
    place_details = gmaps.place(place_id=place_id, fields=['name', 'review'])
    company_name = place_details.get('result', {}).get('name', 'Unknown Company')
    reviews = place_details.get('result', {}).get('reviews', [])
    
    # Filter and transform reviews
    review_data = []
    for review in reviews:
        review_time = datetime.fromtimestamp(review['time'])
        if review_time.year >= datetime.now().year - 5:  # Filter reviews from the last ... years
            review_data.append({
                'company_name': companyc_name,
                'text_review': review['text'],
                'ranking': review['rating'],
                'year': review_time.year
            })
    return review_data

def search_company(city, keyword, radius=50000): # Adjust radius as needed
    # Use geocoding API to get coordinates of the city
    geocode_result = gmaps.geocode(city)
    if not geocode_result:
        return []
    city_location = geocode_result[0]['geometry']['location']
    # Search for places with the specified keyword in the city
    search_result = gmaps.places_nearby(location=city_location, keyword=keyword, radius=radius)  
    return [place['place_id'] for place in search_result.get('results', [])]



place_ids = search_company("Regensburg", "Abteilung Ausländerangelegenheiten")
# print('Result', place_ids)
for place_id in place_ids:
    place_name = get_place_details(place_id, api_key=key)
    # print(place_name)

all_reviews = []
for pid in place_ids:
    all_reviews.extend(fetch_reviews(pid))

df = pd.DataFrame(all_reviews)
print(df.head)
df.to_csv('ausland_reviews.csv', index=False)
