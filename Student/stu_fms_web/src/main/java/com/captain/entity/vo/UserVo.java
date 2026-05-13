package com.captain.entity.vo;

import com.captain.entity.po.Role;
import com.captain.entity.po.User;
import lombok.Data;

@Data
public class UserVo extends User {
    private Long roleId;
    private String roleName;
}
