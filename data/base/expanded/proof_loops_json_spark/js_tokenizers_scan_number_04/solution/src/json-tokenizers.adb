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

package body JSON.Tokenizers
  with SPARK_Mode => On
is

   -------------------------
   -- Scan_Number_Scale_Up --
   -------------------------

   procedure Scan_Number_Scale_Up
     (F_Val : in out Base_Float; Scale : Integer; Status : out Token_Status)
     is separate;

end JSON.Tokenizers;
