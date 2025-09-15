/* 
Calculates the Top k most streamed songs in each country 
*/
SET @topk=5; -- We are getting the top 5 songs in each country

WITH TABLE_A AS (
	SELECT 
		track_title, 
		artist_name, 
		genre, 
		stream_count, 
		streams.country AS country,
		artists.country as artist_country
	FROM tracks
	INNER JOIN artists ON tracks.artist_id = artists.artist_id
	INNER JOIN streams ON streams.track_id = tracks.track_id
),
TABLE_B AS (
	SELECT 
		*,
		DENSE_RANK() OVER (PARTITION BY country ORDER BY stream_count DESC) AS "rank"
	FROM TABLE_A
)
SELECT * FROM TABLE_B WHERE TABLE_B.rank <= @topk
ORDER BY country DESC, TABLE_B.rank ASC;