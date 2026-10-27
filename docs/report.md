## 1. Zero-shot vs Few-shot

| # | Input 			      | Zero-shot raw output      	       | Few-shot raw output | Correct    | Zero OK?  | Few OK?   |
|---|---    			      |---                   		       |---		     |---	  |---	      |---	  |
| 0 | Oh great, another delay...      | Negative + Reasoning (sarcasm) 	       | Negative 	     | Negative   | Yes       | Yes       |
| 1 | The battery lasts forever...    | Negative (or Neutral...) + explanation | Positive            | Mixed      | Hedged    | No        |
| 2 | The package arrived on Tuesday. | Neutral + Reasoning		       | Neutral             | Neutral    | Yes       | Yes       |
| 3 | I wouldn't say it's bad...      | Neutral (leaning slightly Negative)    | Neutral             | Ambiguous  | Acceptable | Acceptable|
| 4 | Best purchase I've made...      | Positive 			       | Positive            | Positive   | Yes       | Yes       |


## 2. Chain-of-Thought

Model: Gemini 2.5 Flash, Temperature: 0, a new chat for every run.

Direct prompt format: `<problem>` + "Give only the final answer, with no explanation."
CoT prompt format: `<problem>` + "Let's think step by step, and then give the final answer on the last line."

### Problem 1 (pens)

Problem:
```
A shop sells pens at 12 rupees each. Ravi buys 7 pens and pays with a 100 rupee note. The shopkeeper gives a 10% discount on the total pen price, but charges 5 rupees for a bag. How much change does Ravi get back?
```
Correct answer: 19.4 rupees

Direct raw output:
```
19.4 rupees
```

CoT raw output:
```
Here is the step-by-step calculation:

1. Calculate the original cost of the pens:

Cost=7 pens×12 rupees=84 rupeesCost=7 pens×12 rupees=84 rupees

2. Calculate the 10% discount on the pens:

Discount=10% of 84=0.10×84=8.4 rupeesDiscount=10% of 84=0.10×84=8.4 rupees

3. Calculate the price of the pens after the discount:

Discounted Pen Price=84−8.4=75.6 rupeesDiscounted Pen Price=84−8.4=75.6 rupees

4. Add the charge for the bag to find the total bill:

Total Bill=75.6+5=80.6 rupeesTotal Bill=75.6+5=80.6 rupees

5. Calculate the change Ravi gets back from 100 rupees:

Change=100−80.6=19.4 rupeesChange=100−80.6=19.4 rupees

19.4 rupees (or 19 rupees and 40 paise)
```

Direct correct? Yes. CoT correct? Yes.

### Problem 2 (train)

Problem:
```
A train leaves Pune at 9:15 AM travelling at 60 km/h. A second train leaves the same station at 10:00 AM on the same track travelling at 80 km/h. At what time does the second train catch up with the first?
```
Correct answer: 12:15 PM

Direct raw output:
```
12:15 PM
```

CoT raw output:
```
To find the time at which the second train catches up with the first, we can follow these steps:

1. Calculate the head start time of the first train:
   * The first train departs at 9:15 AM.
   * The second train departs at 10:00 AM.
   * Head start =

10:00 AM−9:15 AM=45 minutes=4560 hours=0.75 hours10:00 AM−9:15 AM=45 minutes=6045 hours=0.75 hours

2. Calculate the distance covered by the first train during the head start:
   * Distance=Speed×Time=60 km/h×0.75 hours=45 kmDistance=Speed×Time=60 km/h×0.75 hours=45 km
   * At 10:00 AM, the first train is 45 km45 km ahead.

3. Determine the relative speed:
   * Since both trains are traveling in the same direction:
   * Relative Speed=80 km/h−60 km/h=20 km/hRelative Speed=80 km/h−60 km/h=20 km/h

4. Calculate the time required for the second train to catch up:
   * Time=DistanceRelative Speed=45 km20 km/h=2.25 hoursTime=Relative SpeedDistance=20 km/h45 km=2.25 hours
   * 2.25 hours=2 hours and (0.25×60) minutes=2 hours and 15 minutes2.25 hours=2 hours and (0.25×60) minutes=2 hours and 15 minutes

5. Find the catch-up time:
   * Adding this duration to the departure time of the second train:
   * 10:00 AM+2 hours and 15 minutes=12:15 PM10:00 AM+2 hours and 15 minutes=12:15 PM

12:15 PM
```

Direct correct? Yes. CoT correct? Yes.

Note: the doubled text in the math lines is a copy artifact from LaTeX rendering in AI Studio, not part of the model's actual answer.

### Analysis

On both problems the direct prompt and the step-by-step prompt gave the correct
final answer (19.4 rupees and 12:15 PM). So chain-of-thought did not improve
accuracy here. Gemini 2.5 Flash appears to reason internally before answering
(the "Thoughts" section in AI Studio), so the direct prompt already benefits
from step-by-step thinking.

What CoT did change was visibility: the step-by-step output shows the head
start, the gap, the relative speed and the final time, so each answer can be
checked line by line, while the direct answer gives only the result. CoT also
made the output much longer.

Limitation: I tested only two problems, one run each, so these results do not
show that CoT never helps. It may matter more on harder problems or smaller
models.


### Version 1 (no schema): raw outputs

(inputs 1-4 and 6 from earlier, input 5 below)

Input 5:
```json
{
  "name": "Kavita Joshi",
  "date": "August 1, 2026",
  "amount": "$120"
}
```

### Version 2 (explicit schema + rules): raw outputs

Input 1:
```json
{"name": "Priya Sharma", "date": "2026-03-03", "amount": 2450}
```
Input 2:
```json
{"name": "Mr. Anil Kulkarni", "date": "2026-04-12", "amount": 15000}
```
Input 3:
```json
{"name": "Sunita Patil", "date": "2026-01-05", "amount": 7500.5}
```
Input 4:
```json
{"name": "Rohan Deshmukh", "date": "2026-06-14", "amount": 3000}
```
Input 5:
```json
{"name": "Kavita Joshi", "date": "2026-08-01", "amount": 120}
```
Input 6:
```json
{"name": null, "date": null, "amount": null}
```

### Analysis (V1 vs V2)

V1 returned valid JSON every time, but only 1/6 outputs matched the target
schema: dates stayed in the original text format and amounts were strings with
currency text. Only input 6 (no data) matched, because all fields were null.
V2 matched the schema on 6/6 inputs: dates in YYYY-MM-DD, amounts as numbers,
no code fences or extra text. Stating the exact schema and format rules was
the biggest improvement.

V2 was not perfect. On input 2 the title "Mr." stayed in the name, and the
ambiguous date 12/04/2026 was read as DD/MM without being flagged. On input 4
the vague "around 3 thousand" became exactly 3000 even though the prompt said
not to guess. On input 5 the "$" was dropped because the schema has no
currency field. Strictly counted, 4/6 outputs were fully correct without caveats.


### Version 3 (schema + 2 examples + Structured output): raw outputs

Input 1:
```json
{"name": "Priya Sharma", "date": "2026-03-03", "amount": 2450}
```
Input 2:
```json
{"name": "Anil Kulkarni", "date": "2026-04-12", "amount": 15000}
```
Input 3:
```json
{"name": "Sunita Patil", "date": "2026-01-05", "amount": 7500.5}
```
Input 4:
```json
{"name": "Rohan Deshmukh", "date": "2026-06-14", "amount": 3000}
```
Input 5:
```json
{"name": "Kavita Joshi", "date": "2026-08-01", "amount": 120}
```
Input 6:
```json
{"name": null, "date": null, "amount": null}
```

### Results

| Input | V1 matches schema? | V2 matches schema? | V3 matches schema? | Notes |
|---|---|---|---|---|
| 1 | No | Yes | Yes | |
| 2 | No | Yes | Yes | V2 kept "Mr." in name; V3 dropped it. DD/MM assumed in both |
| 3 | No | Yes | Yes | |
| 4 | No | Yes | Yes | "around 3 thousand" became 3000 in V2 and V3 |
| 5 | No | Yes | Yes | "$" dropped, schema has no currency field |
| 6 | Yes | Yes | Yes | all null |

Schema match: V1 1/6, V2 6/6, V3 6/6.
Fully correct with no caveats (DD/MM assumed): V1 1/6, V2 4/6, V3 5/6.

### Analysis

All three versions produced valid JSON, so validity was never the problem.
The real problem in V1 was format: dates and amounts were copied from the text
as strings ("3rd March 2026", "Rs. 2,450"), so only 1/6 outputs matched the
target schema. Stating the schema and format rules in V2 fixed this (6/6 schema
matches); that was the largest single improvement.

V3 added two examples and Structured output. It matched the schema 6/6 as V2
did, and fixed one detail: the title "Mr." was no longer included in the name
on input 2. I changed two things at once (examples and Structured output), so I
cannot say which one caused this; it may simply be run-to-run variation.

Nothing fixed input 4: "around 3 thousand rupees" was turned into exactly 3000
in V2 and V3, even though the prompt said not to guess. Input 5 lost the
currency ("$") because my schema has no currency field. The ambiguous date
12/04/2026 was read as DD/MM without any warning. Also, my Structured output
schema did not allow null, yet input 6 still returned null; I am not sure the
schema was enforced strictly in that run.
Limitation: 6 inputs, one run each.

---

## 4. Key Takeaways

- Few-shot examples made output format consistent (one word), but did not fix accuracy on a mixed-sentiment review because none of my examples covered that case.
- Chain-of-thought did not change accuracy on my two problems, but made the reasoning visible and checkable.
- An explicit schema with format rules was the biggest improvement for JSON (1/6 to 6/6 schema matches).
- Ambiguous inputs (vague amounts, DD/MM dates, missing currency) stay hard even with a good prompt, and need explicit handling rules or extra schema fields.