--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

--  Extracted from json-spark, json/src/json-tokenizers.ads and .adb.
--
--  Base_Float and Mantissa_Limit are upstream's, declared inside Scan_Number's
--  body; they had to move to the spec here so the profile and the contract can
--  name them.  Power_Limit and P stay where upstream put them, in the declare
--  block that encloses the loop.

with JSON.Types;

generic
   with package Types is new JSON.Types (<>);
package JSON.Tokenizers with SPARK_Mode => On is

   pragma Preelaborate;

   use type Types.Float_Type;
   --  Upstream's body carries this clause; the declarations and the contract
   --  below need it in the visible part

   subtype Base_Float is Types.Float_Type'Base;

   Mantissa_Limit : constant Base_Float := Base_Float'Last / 16.0;
   --  Base_Float'Last / 16.0 is exact (division by a power of two),
   --  which makes the overflow guards below provable

   procedure Scan_Number_Scale_Down
     (F_Val : in out Base_Float; Scale : Integer)
   with
     Pre  =>
       F_Val in 0.0 .. Mantissa_Limit * 10.0 + 9.0
       and then Scale > Integer'First,
     Always_Terminates,
     Post => F_Val in 0.0 .. Mantissa_Limit * 10.0 + 9.0;
   --  Divide the accumulated float mantissa by ten -Scale times, in as few
   --  roundings as the exactly-representable range allows.
   --
   --  The bound on F_Val is the one Add_Digit maintains on the accumulator
   --  F_Val is copied from.  Scale > Integer'First is what makes the loop
   --  bound -Scale well-defined; upstream needs no such contract because it
   --  computes Scale from the exponent and the fraction digit count, whose
   --  ranges it knows.  It is the weakest form of that fact: a bound naming
   --  the exponent limit would exclude configurations this loop handles
   --  perfectly well.

end JSON.Tokenizers;
