# Completing Unfinished Bach
## From music21's Bach's chorale corpus to an explorable DataFrame

<br>One of the safest and cleanest ways (without any noise) to obtain Bach's chorales into a dataset is through music21.
Music21 is a well-known trustworthy library that contains songs for the purpose of being studied, with useful information with more than
just the partiture scores tuned into data (Roman number-chords, MIDI, anacrusis, offset, etc.).

However, the data exists in the form of nested stream structures that need careful processing, in order to create a readable
and understandable structure.
<br><br>
### **Step 1: Bach's chorales in music21's corpus.**
To do that, there are different ways. First, there is the method displayed music 21's
[User’s Guide, Chapter 11: Corpus Searching](https://music21.org/music21docs/usersGuide/usersGuide_11_corpusSearching.html)
as a general example through corpus search corpus.search('Bach', 'composer'). We look through the corpus collection 
and bring back all of the songs that Bach is their composer. The second option is corpus.chorales.Iterator(), an iterator
for the Bach chorale corpus. After checking how it works and its contents, the conclusion was that it just limits the 
data that corpus.search provides, but it will be used in order to create the validation set later. There are other methods also, like local corpus (we do not have chorales to upload),
and obtaining the data online through music21 (this method introduces noise and additional complexity while we already have accurately all of Bach's chorales with other methods).
<br>
### **Step 2: Removing Duplicate titles and detecting different harmonic interpretations on the same song**
After obtaining the dataset, it is essential to split the duplicate songs from the different harmonic interpretation
and spot them, in order to use most of the examples, without any data leaks. 

We will look based on the BWV ids (chorale's number) by extracting the numbers only and in between dots from
the song titles. After that, we will look for the duplicated BWV ids. Now, from the duplicated IDs will run the
duplicated algorithm but for their titles. The unique examples left are the titles that have not been duplicated.

### **Step 3: Organizing the data into a form that will be searchable and useful by TensorFlow models.**

 - Analyzing a Baroque song, starts by recognizing the key. The key can be obtained by using the function analyze('key').
 - Before transferring the data from an object to a data frame, the names of the columns (variables) must be decided as well as a way/names of the data that will be extracted from the object.
 - Now, the data frame will be used as time-series and being able to navigate using the song's bars and offset is essential. So by using the function getElementsByClass('Measure') time-based data will become available.
 - The names of the data that the data frame will use based on the measure are "number", "numberSuffix", "offset", "duration.quarterLength" and "paddingLeft" | referring to | "bar", "suffix", "offset", "len" and "anacrusis"
 - In order to prove our ability to use/navigate through the dataset and create it, we must first be able to provide a representation of the song with variables per bar for each SATB (Soprano-Alto-Tenor-Bass) voice.
 - This has been achieved by navigating using the bar number and the suffix (if it is not empty). Be careful when the suffix is not null, the bar number is repeated, which means that you need to navigate using the unique bar numbers and afterwards check for suffix. Last but not least, this exact way for searching must be used to print the chorale the way we mentioned earlier. Once that is done, the ability to manipulate the object to our will has been obtained and the dataset can be created.

Example of printing bwv101.7:

```
current bar: 0
Soprano
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.clef.TrebleClef>
{0.0} <music21.key.Key of d minor>
{0.0} <music21.meter.TimeSignature 4/4>
{0.0} <music21.note.Note A>
Alto
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.clef.TrebleClef>
{0.0} <music21.key.Key of F major>
{0.0} <music21.meter.TimeSignature 4/4>
{0.0} <music21.note.Note F>
Tenor
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.clef.BassClef>
{0.0} <music21.key.Key of F major>
{0.0} <music21.meter.TimeSignature 4/4>
{0.0} <music21.note.Note D>
Bass
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.clef.BassClef>
{0.0} <music21.key.Key of F major>
{0.0} <music21.meter.TimeSignature 4/4>
{0.0} <music21.note.Note D>
current bar: 1
Soprano
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note F>
{2.0} <music21.note.Note G>
{3.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note E>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note D>
Tenor
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note B->
{3.0} <music21.note.Note A>
Bass
{0.0} <music21.note.Note C#>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note B->
{3.0} <music21.note.Note F>
current bar: 2
Soprano
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note D>
{1.0} <music21.note.Note C#>
{2.0} <music21.note.Note A>
{3.0} <music21.note.Note F>
Tenor
{0.0} <music21.note.Note B->
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note D>
Bass
{0.0} <music21.note.Note G>
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note D>
{3.5} <music21.note.Note E>
current bar: 3
Soprano
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note G>
{1.5} <music21.note.Note F>
{2.0} <music21.note.Note E>
{3.0} <music21.note.Note F>
Tenor
{0.0} <music21.note.Note C>
{1.0} <music21.note.Note C>
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note C>
Bass
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note A>
{3.0} <music21.note.Note F>
current bar: 4
Soprano
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note D>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note C#>
Tenor
{0.0} <music21.note.Note B->
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note E>
Bass
{0.0} <music21.note.Note B->
{1.0} <music21.note.Note B->
{2.0} <music21.note.Note A>
current bar: 4a
Soprano
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.note.Note A>
Alto
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.note.Note D>
Tenor
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.note.Note F>
Bass
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.note.Note D>
current bar: 5
Soprano
{0.0} <music21.note.Note C>
{1.0} <music21.note.Note D>
{1.5} <music21.note.Note E>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note E>
Alto
{0.0} <music21.note.Note E>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note A>
{3.0} <music21.note.Note G>
Tenor
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note B->
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note B->
{3.5} <music21.note.Note C>
Bass
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note G>
{3.5} <music21.note.Note A>
current bar: 6
Soprano
{0.0} <music21.note.Note D>
{1.0} <music21.note.Note C#>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note D>
Alto
{0.0} <music21.note.Note F>
{0.5} <music21.note.Note G>
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note A>
{3.0} <music21.note.Note G>
Tenor
{0.0} <music21.note.Note D>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note B>
Bass
{0.0} <music21.note.Note B->
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note G>
current bar: 7
Soprano
{0.0} <music21.note.Note E>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note B>
Alto
{0.0} <music21.note.Note G>
{1.0} <music21.note.Note F#>
{1.5} <music21.note.Note G#>
{2.0} <music21.note.Note A>
{3.0} <music21.note.Note E>
Tenor
{0.0} <music21.note.Note C>
{0.5} <music21.note.Note B>
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note D>
Bass
{0.0} <music21.note.Note C>
{1.0} <music21.note.Note D>
{1.5} <music21.note.Note E>
{2.0} <music21.note.Note F#>
{3.0} <music21.note.Note G#>
current bar: 8
Soprano
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note B>
{2.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note E>
{0.5} <music21.note.Note A>
{1.0} <music21.note.Note A>
{1.5} <music21.note.Note G#>
{2.0} <music21.note.Note E>
Tenor
{0.0} <music21.note.Note C>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note C>
Bass
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note A>
current bar: 8a
Soprano
{0.0} <music21.layout.PageLayout>
{0.0} <music21.layout.SystemLayout>
{0.0} <music21.note.Note D>
Alto
{0.0} <music21.layout.PageLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.note.Note D>
Tenor
{0.0} <music21.layout.PageLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.note.Note G>
Bass
{0.0} <music21.layout.PageLayout>
{0.0} <music21.layout.StaffLayout distance 130, staffNumber 1, staffSize None, staffLines None>
{0.0} <music21.note.Note B>
current bar: 9
Soprano
{0.0} <music21.note.Note C>
{1.0} <music21.note.Note B>
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note E>
{1.0} <music21.note.Note F>
{2.0} <music21.note.Note G>
{3.0} <music21.note.Note F>
Tenor
{0.0} <music21.note.Note G>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note C>
Bass
{0.0} <music21.note.Note C>
{1.0} <music21.note.Note D>
{2.0} <music21.note.Note E>
{3.0} <music21.note.Note F>
current bar: 10
Soprano
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note A>
Alto
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note C>
{3.0} <music21.note.Note D>
Tenor
{0.0} <music21.note.Note D>
{1.0} <music21.note.Note G>
{2.0} <music21.note.Note A>
{3.0} <music21.note.Note A>
Bass
{0.0} <music21.note.Note B->
{1.0} <music21.note.Note C>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note F#>
current bar: 11
Soprano
{0.0} <music21.note.Note B->
{1.0} <music21.note.Note A>
{1.5} <music21.note.Note G>
{2.0} <music21.note.Note F>
{3.0} <music21.note.Note G>
Alto
{0.0} <music21.note.Note D>
{1.0} <music21.note.Note C#>
{2.0} <music21.note.Note D>
{3.0} <music21.note.Note E>
Tenor
{0.0} <music21.note.Note G>
{0.5} <music21.note.Note F>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note B->
Bass
{0.0} <music21.note.Note G>
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note B->
{3.0} <music21.note.Note A>
{3.5} <music21.note.Note G>
current bar: 12
Soprano
{0.0} <music21.note.Note F>
{1.0} <music21.note.Note E>
{2.0} <music21.note.Note D>
{3.0} <music21.bar.Barline type=final>
Alto
{0.0} <music21.note.Note C#>
{0.5} <music21.note.Note D>
{1.5} <music21.note.Note C#>
{2.0} <music21.note.Note A>
{3.0} <music21.bar.Barline type=final>
Tenor
{0.0} <music21.note.Note B->
{0.5} <music21.note.Note A>
{1.0} <music21.note.Note G>
{1.5} <music21.note.Note A>
{2.0} <music21.note.Note F#>
{3.0} <music21.bar.Barline type=final>
Bass
{0.0} <music21.note.Note A>
{1.0} <music21.note.Note A>
{2.0} <music21.note.Note D>
{3.0} <music21.bar.Barline type=final>
```
                    
                     



