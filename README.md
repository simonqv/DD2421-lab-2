# Answers

## 1
Till slut kommer vi till en punkt där vi inte kan hitta en lösning längre och så hamnar vår avgränsning på sidan (typ lite random) då en inte kan dra någon vettig linje mellan grupperna. (Vi får hög bias)

## 2
Se biler/PDFer

## 3
- Polynomial: Vid låga värden på `p` (typ 1) så har vi ganska hög bias och låg varians i våran lösning (vi får typ en rak linje som går igenom all data), men när vi ökar `p` så minskar vår bias medans vår varians ökar (Vi fångar fler datapunker med en mer komplicerad avgränsning).
- Radialbasis: Vid höga värden på `sig` (typ 10) så har vi ganska hög bias och låg varians i våran lösning (vi får typ en rak linje som går igenom all data), men när vi minskar `sig` så minskar vår bias medans vår varians ökar (Vi fångar fler datapunker med en mer komplicerad avgränsning).

## 4
För små värden så tillåter vi flera värden som är fel, vilket kan leda till att vi klasifierar alla som samma sak. Vid väldigt höga värden tillåter vi inte längre några fel och då får vi att ingen punkt ligger inom marginalerna.

## 5
Om vi misstänker mycket "noise" i vår data så kan det vara en bra idé att tillåta mer slack för att hantera bruset. Om vi istället misstänker att datan är för komplex för vår nuvarande modell så bör vi ta en mer komplicerad modell. Det kan också vara en idé att låta det vara mer slack istället för att öka komplexiteten på modellen för att undvika overfitting och därmed få en mer generellt applicerbar modell.
