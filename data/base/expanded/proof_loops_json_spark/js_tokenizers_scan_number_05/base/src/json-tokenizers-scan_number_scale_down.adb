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

separate (JSON.Tokenizers)
procedure Scan_Number_Scale_Down
  (F_Val : in out Base_Float; Scale : Integer) is
begin
   if F_Val /= 0.0 then
      if Scale < 0 then
         --  The divisor is built up exactly and applied in a
         --  single division where possible, so that small
         --  decimal numbers like 3.14 convert with a single
         --  rounding. Power_Limit is exact (a division by a
         --  power of two), which makes the guards provable.
         declare
            Power_Limit : constant Base_Float := Base_Float'Last / 256.0;
            P           : Base_Float := 1.0;
         begin
            for J in 1 .. -Scale loop
               if P > Power_Limit then
                  F_Val := F_Val / P;
                  P := 1.0;
               end if;
               P := P * 10.0;
            end loop;
            F_Val := F_Val / P;
         end;
      end if;
   end if;
end Scan_Number_Scale_Down;
