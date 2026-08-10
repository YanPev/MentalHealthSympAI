# Data Availability Report — Week 1

## Dataset location

Project directory:

`/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project`

Transcript directory:

`/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project/data/raw/edaic/transcripts`

Labels directory:

`/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project/data/raw/edaic/labels`

## Transcript availability

Number of transcript files:

275

Example transcript file:

`300_Transcript.csv`

Transcript columns from sample:

['Start_Time', 'End_Time', 'Text', 'Confidence']

## PHQ-8 label availability

Detailed PHQ-8 labels file:

`/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project/data/raw/edaic/labels/labels/Detailed_PHQ8_Labels.csv`

Number of rows in labels file:

219

Participant ID column:

`Participant_ID`

PHQ-8 item columns:

['PHQ_8NoInterest', 'PHQ_8Depressed', 'PHQ_8Sleep', 'PHQ_8Tired', 'PHQ_8Appetite', 'PHQ_8Failure', 'PHQ_8Concentrating', 'PHQ_8Moving']

## Participant overlap

| check                         |   count |
|:------------------------------|--------:|
| participants_with_transcript  |     275 |
| participants_with_phq8_labels |     219 |
| participants_with_both        |     219 |
| transcript_only               |      56 |
| labels_only                   |       0 |

Valid participant IDs file:

`/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project/outputs/valid_participant_ids.csv`

## Main conclusion

The dataset contains transcript files and PHQ-8 item-level labels. The valid participant pool for the next stage is the overlap between participants with transcripts and participants with PHQ-8 labels.

## Next steps

- Build PHQ-8 item-level dataset.
- Create participant-level train/dev/test split.
- Build utterance bank from the transcript `Text` column.
