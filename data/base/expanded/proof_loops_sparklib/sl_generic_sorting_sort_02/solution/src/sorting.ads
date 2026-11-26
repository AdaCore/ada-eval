--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

--  Generic_Sorting.Sort's heap-extraction loop, isolated.
--
--  Sort is kept in its upstream shape: the same profile, the same contract,
--  the same declarative part, less its nested Sift.  Sift is reduced to an
--  imported declaration carrying upstream's own Sift contract, declared in
--  the package body so that Sort's body is a subunit of its own, and so takes
--  the Sort objects it uses as parameters.

pragma Ada_2022;

with SPARK.Containers.Types; use SPARK.Containers.Types;
with Vectors; use Vectors;

package Sorting with SPARK_Mode, Always_Terminates is

   procedure Sort (Container : in out Vector)
   with
     Global => null,
     Post   => (Static => Length (Container) = Length (Container)'Old);

end Sorting;
