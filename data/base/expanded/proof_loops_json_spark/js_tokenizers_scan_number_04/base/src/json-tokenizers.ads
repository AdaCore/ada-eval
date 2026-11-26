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
--  name them.

with JSON.Types;

generic
   with package Types is new JSON.Types (<>);
package JSON.Tokenizers with SPARK_Mode => On is

   pragma Preelaborate;

   use type Types.Float_Type;
   --  Upstream's body carries this clause; the declarations and the contract
   --  below need it in the visible part

   type Token_Status is
     (Success,
      Error_Unexpected_EOF,
      Error_Expected_EOF,
      Error_Unexpected_Character,
      Error_Control_Character,
      Error_Escaped_Character,
      Error_Escaped_Unicode,
      Error_Plus_Sign,
      Error_Leading_Zeroes,
      Error_Minus_Digit,
      Error_Dot_Digit,
      Error_Exponent_Digit,
      Error_Exponent_Sign_Digit,
      Error_Number_Too_Long,
      Error_Number_Out_Of_Range,
      Error_Invalid_Literal);
   --  The outcome of scanning a token: Success or a specific error code

   subtype Base_Float is Types.Float_Type'Base;

   Mantissa_Limit : constant Base_Float := Base_Float'Last / 16.0;
   --  Base_Float'Last / 16.0 is exact (division by a power of two),
   --  which makes the overflow guards below provable

   procedure Scan_Number_Scale_Up
     (F_Val : in out Base_Float; Scale : Integer; Status : out Token_Status)
   with
     Pre  => F_Val in 0.0 .. Mantissa_Limit * 10.0 + 9.0,
     Always_Terminates,
     Post => F_Val in 0.0 .. Mantissa_Limit * 10.0 + 9.0;
   --  Multiply the accumulated float mantissa by ten Scale times, reporting
   --  Error_Number_Out_Of_Range if that would leave the range of Base_Float.

end JSON.Tokenizers;
