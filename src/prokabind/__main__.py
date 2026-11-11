"""`python -m prokabind run|show …` — программа ProkaBind (SPEC раздел 5); см. `prokabind.program`."""
import sys

from .program import main

sys.exit(main())
