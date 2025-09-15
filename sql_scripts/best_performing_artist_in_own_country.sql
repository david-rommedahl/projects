/*
This script returns the artist(s) with the most songs in the toplist of their own country
*/

SET @top_k = 10; -- We are interested in the top 10 songs of each country

WITH TABLE_A AS (
	SELECT 
	track_title,
	album_title,
	artist_name,
	genre,
	stream_count,
	artists.country AS artist_country,
	streams.country AS country
	FROM
	artists
	INNER JOIN tracks ON artists.artist_id=tracks.artist_id
	INNER JOIN streams ON streams.track_id=tracks.track_id
),
TABLE_B AS (
	SELECT 
	*,
	DENSE_RANK() OVER (PARTITION BY country ORDER BY stream_count DESC) AS "rank"
	FROM TABLE_A
),
TABLE_C AS (
	SELECT artist_name, artist_country, COUNT(*) AS top_song_count
	FROM TABLE_B
	WHERE TABLE_B.rank <= @top_k AND artist_country = country 
	GROUP BY artist_name, artist_country
	ORDER BY top_song_count DESC
)
SELECT * FROM TABLE_C
WHERE top_song_count = (SELECT MAX(top_song_count) FROM TABLE_C);