--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;

package body Lists with SPARK_Mode
is

   procedure Move (Target : in out List; Source : in out List) is separate;

end Lists;
