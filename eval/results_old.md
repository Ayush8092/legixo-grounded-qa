# Evaluation results

- Total questions: 16
- Passed (all three checks): 6/16
- Abstention correctness: 6/16
- Answer correctness: 6/16
- Citation correctness: 6/16

| ID | Category | Answerable? | Found? | Pass | Question |
|---|---|---|---|---|---|
| eval-01 | answerable_direct | True | True | ✅ | What is the notice period at Bluecrest Analytics? |
| eval-02 | answerable_semantic | True | True | ✅ | How long is the non-compete period after leaving Bluecrest, and what triggers it? |
| eval-03 | answerable_numeric | True | True | ✅ | What is the monthly rent for Unit 4B at Harbor View Tower? |
| eval-04 | answerable_numeric | True | False | ❌ | How much is the security deposit on the Harbor View lease, and how many months' rent does that represent? |
| eval-05 | answerable_direct | True | False | ❌ | Is subletting allowed under the Harbor View lease? |
| eval-06 | answerable_numeric | True | False | ❌ | What interest rate applies to delayed payments under the Riverside Commercial Courts Act when the contract fixes no rate? |
| eval-07 | answerable_numeric | True | False | ❌ | When does a commercial suit have to go to mandatory mediation under the Riverside Commercial Courts Act, and for how long? |
| eval-08 | answerable_numeric | True | False | ❌ | What did Northfield offer to settle the Arvind Mehta case, and what did the client instruct as a counter? |
| eval-09 | multi_document | True | False | ❌ | When is the next hearing in Arvind Mehta v. Northfield Logistics, and what is scheduled to happen at it? |
| eval-10 | answerable_direct | True | False | ❌ | How many days before a hearing must written arguments be filed? |
| eval-11 | answerable_semantic | True | False | ❌ | What counts as confidential information under the Bluecrest employment agreement? |
| eval-12 | similar_terminology | True | False | ❌ | What is the limitation period for contract claims under the fictional Riverside Code mentioned in the Arvind Mehta memo? |
| eval-13 | unrelated | False | False | ✅ | What is the population of Riverside city? |
| eval-14 | out_of_corpus | False | False | ✅ | Who is the president of India? |
| eval-15 | out_of_corpus | False | False | ✅ | What color is the carpet in the Harbor View Tower office? |
| eval-16 | partial_information | True | False | ❌ | What penalty applies if Priya breaches the non-compete clause at Bluecrest? |

## Details

### eval-01 — PASS (3.86s)
- Question: What is the notice period at Bluecrest Analytics?
- Answer: The notice period at Bluecrest Analytics is 60 days.
- Citations: ['02_employment_agreement_excerpt::notice-period::0']
- Checks: abstention=True answer=True citation=True

### eval-02 — PASS (1.95s)
- Question: How long is the non-compete period after leaving Bluecrest, and what triggers it?
- Answer: The non-compete period is 12 months after leaving Bluecrest, triggered if the new role is with a direct competitor in the same city where Bluecrest has an office and uses the same client list supplied by Bluecrest.
- Citations: ['02_employment_agreement_excerpt::non-compete::0']
- Checks: abstention=True answer=True citation=True

### eval-03 — PASS (2.11s)
- Question: What is the monthly rent for Unit 4B at Harbor View Tower?
- Answer: The monthly rent for Unit 4B at Harbor View Tower is ₹45,000.
- Citations: ['06_property_lease_clause::rent-and-deposit::0']
- Checks: abstention=True answer=True citation=True

### eval-04 — FAIL (1.48s)
- Question: How much is the security deposit on the Harbor View lease, and how many months' rent does that represent?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-05 — FAIL (3.61s)
- Question: Is subletting allowed under the Harbor View lease?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-06 — FAIL (3.58s)
- Question: What interest rate applies to delayed payments under the Riverside Commercial Courts Act when the contract fixes no rate?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-07 — FAIL (3.52s)
- Question: When does a commercial suit have to go to mandatory mediation under the Riverside Commercial Courts Act, and for how long?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-08 — FAIL (3.91s)
- Question: What did Northfield offer to settle the Arvind Mehta case, and what did the client instruct as a counter?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-09 — FAIL (4.5s)
- Question: When is the next hearing in Arvind Mehta v. Northfield Logistics, and what is scheduled to happen at it?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-10 — FAIL (4.56s)
- Question: How many days before a hearing must written arguments be filed?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-11 — FAIL (4.62s)
- Question: What counts as confidential information under the Bluecrest employment agreement?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-12 — FAIL (4.14s)
- Question: What is the limitation period for contract claims under the fictional Riverside Code mentioned in the Arvind Mehta memo?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False

### eval-13 — PASS (3.3s)
- Question: What is the population of Riverside city?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=True answer=True citation=True

### eval-14 — PASS (3.2s)
- Question: Who is the president of India?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=True answer=True citation=True

### eval-15 — PASS (2.83s)
- Question: What color is the carpet in the Harbor View Tower office?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=True answer=True citation=True

### eval-16 — FAIL (2.8s)
- Question: What penalty applies if Priya breaches the non-compete clause at Bluecrest?
- Answer: I cannot find the answer to this question in the provided documents.
- Citations: (none)
- Checks: abstention=False answer=False citation=False
