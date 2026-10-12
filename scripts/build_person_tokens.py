"""Build data/person_name_tokens.txt — the word list behind the classifier's
person-name guard (app/services/merchant_classifier.py).

    python scripts/build_person_tokens.py <CreditMitra-Merchant-Classification>/data

Every word of length >= 3 from the labelled Indian person-name corpus, plus
common first names and surnames the corpus happens to miss, minus business
words that the generated corpus lets slip in ("tailor", "bag"). A payee is
treated as a person only if *every* word in it is on this list, so a shop
name ("Ganesh Traders") never matches — it contains a business word.

Held-out check (80/20 split of both corpora): 99.3% of person names
recognised, 3.4% of merchant names flipped — almost all of those are bare
owner names ("Rajesh K..", "Gupta ..") that no name-based rule can tell from
a person, and for which "person" is the privacy-safe reading.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "person_name_tokens.txt"

# Common names missing from the generated corpus.
EXTRA_NAMES = """
aarav aarti abhay abhinav abhishek aditi aditya ajay akash akhil akshay alok aman amar amit
amita amrita anand anil anita anjali ankit ankita anuj anupam anurag anushka archana arjun
arun aruna arvind asha ashish ashok ashwin ayush bharat bhavna chandan chetan deepa deepak
deepika dev dhruv dinesh divya gaurav geeta girish gopal govind hari harish hemant indu ishaan
jatin jaya jitendra jyoti kajal kamal karan kartik kavita kavya keshav kiran kishore komal
krishna kunal lakshmi lalit lata madhu mahesh manish manisha manoj meena meera mohan mohit
mukesh naresh naveen neha nikhil nisha nitin pallavi pankaj pooja pradeep prakash pranav
prashant pratik praveen preeti priya priyanka puneet rahul raj rajat rajeev rajesh rajiv
rakesh ram ramesh rani ravi reena rekha rishi ritu rohan rohit ruchi sachin sagar sahil
sakshi sameer sandeep sanjay sanjana sarita satish saurabh seema shalini shankar sharad
shashi shilpa shiny shivam shivansh shreya shubham shweta siddharth simran sneha sonal
sonia sudhir sumit sunil sunita suresh swati tanvi tarun tushar uday usha varun vijay vikas
vikram vinay vinod vipin vishal vivek yash yogesh disha gargi
agarwal ahmed ali bansal bhat chauhan chopra das dubey dutta garg ghosh goyal gupta iyer jain
jha joshi kapoor khan khanna kulkarni kumar malhotra mehta menon mishra mukherjee nair pandey
patel pillai rao reddy saxena sen shah sharma shetty singh sinha srivastava tiwari trivedi
verma yadav katiyar kwatra kalia
"""

# Business words the generated corpus lets into person names.
BUSINESS_WORDS = """
academy agency agencies auto bag bags bakery bakers bazaar bhandar cafe center centre classes
cloth clothing coaching collection collections dairy dhaba electric electricals electronics
emporium enterprise enterprises fashion foods footwear fruits furniture garments general hardware
hospital hotel jewel jewellers jewellery kirana kitchen mart medical medicals mobile mobiles
motors pharma pharmacy provision provisions restaurant salon school shop sons store stores
studio sweets tailor tailors textiles tours traders trading travels tutorials
"""

_WORD_RE = re.compile(r"[^a-z ]")


def words(text: str) -> list[str]:
    return [w for w in _WORD_RE.sub(" ", text.lower()).split() if len(w) >= 3]


def build(corpus_dir: Path) -> set[str]:
    with (corpus_dir / "indian_person_names.csv").open(encoding="utf-8") as fh:
        tokens = {w for row in csv.DictReader(fh) for w in words(row["name"])}
    tokens |= set(EXTRA_NAMES.split())
    return tokens - set(BUSINESS_WORDS.split())


def main() -> None:
    tokens = build(Path(sys.argv[1]))
    OUT.write_text("\n".join(sorted(tokens)) + "\n", encoding="utf-8")
    print(f"wrote {len(tokens)} tokens to {OUT}")


if __name__ == "__main__":
    main()
