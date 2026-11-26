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
procedure Scan_Number_Fraction_Digits
  (Stream       : in out Streams.Stream;
   Char_Count   : in out Natural;
   Int_Acc      : in out Base_Integer;
   Int_Overflow : in out Boolean;
   F_Acc        : in out Base_Float;
   F_Overflow   : in out Boolean;
   Frac_Digits  : in out Natural;
   Status       : out Token_Status)
is
   C   : Character;
   EOF : Boolean;

   function To_Digit (Value : Character) return Base_Integer
   is (Character'Pos (Value) - Character'Pos ('0'))
   with Pre => Value in '0' .. '9';

   procedure Add_Digit (Digit : Base_Integer)
   with
     Pre  =>
       Digit in 0 .. 9
       and then Int_Acc <= 0
       and then F_Acc in 0.0 .. Mantissa_Limit * 10.0 + 9.0,
     Always_Terminates,
     Post =>
       Int_Acc <= 0 and then F_Acc in 0.0 .. Mantissa_Limit * 10.0 + 9.0;

   procedure Add_Digit (Digit : Base_Integer) is
   begin
      if not Int_Overflow then
         if Int_Acc < (Base_Integer'First + Digit) / 10 then
            Int_Overflow := True;
         else
            Int_Acc := Int_Acc * 10 - Digit;
         end if;
      end if;

      if not F_Overflow then
         if F_Acc > Mantissa_Limit then
            F_Overflow := True;
         else
            F_Acc := F_Acc * 10.0 + Base_Float (Digit);
         end if;
      end if;
   end Add_Digit;
begin
   Status := Success;

   --  Accept a sequence of digits
   loop
      pragma Loop_Variant (Increases => Streams.Position (Stream));

      Streams.Read_Character (Stream, C, EOF);
      exit when EOF or else C not in '0' .. '9';

      if Char_Count = Types.Maximum_String_Length_Numbers then
         Status := Error_Number_Too_Long;
         return;
      end if;
      Char_Count := Char_Count + 1;

      Frac_Digits := Frac_Digits + 1;
      Add_Digit (To_Digit (C));
   end loop;
end Scan_Number_Fraction_Digits;
