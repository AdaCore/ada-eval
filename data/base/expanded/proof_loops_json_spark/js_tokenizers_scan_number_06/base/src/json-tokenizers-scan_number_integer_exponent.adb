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
procedure Scan_Number_Integer_Exponent
  (Int_Acc   : in out Base_Integer;
   Exp_Value : Natural;
   Status    : out Token_Status) is
begin
   Status := Success;

   if Exp_Value > 0 and Int_Acc /= 0 then
      for J in 1 .. Exp_Value loop
         if Int_Acc < Base_Integer'First / 10 then
            Status := Error_Number_Out_Of_Range;
            return;
         end if;
         Int_Acc := Int_Acc * 10;
      end loop;
   end if;
end Scan_Number_Integer_Exponent;
