#!/bin/bash
# getting rid of the "friends" attribute 
# since it is too long and we may not be able to load it into pandas

cat original_data/yelp_json/yelp_academic_dataset_user.json | jq -c '{
user_id,
name,
review_count, 
yelping_since,
useful,
funny,
cool,
elite,
fans,
average_stars,
compliment_hot,
compliment_more,
compliment_profile,
compliment_cute,
compliment_list,
compliment_note,
compliment_plain,
compliment_cool,
compliment_funny,
compliment_writer,
compliment_photos
}'

