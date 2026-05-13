package com.captain.entity.po;

import com.baomidou.mybatisplus.annotation.*;

import java.util.Date;

import java.io.Serializable;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 *
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class User extends BaseEntity implements Serializable {

    /**
     * 账号
     */
    private String username;

    /**
     * 密码
     */
    private String password;

    /**
     * 昵称
     */
    private String nickName;

    /**
     * 真实姓名
     */
    private String realName;

    /**
     * 邮箱
     */
    private String email;

    /**
     * 手机电话
     */
    private String mobile;

    /**
     * 性别
     */
    private String gender;

    /**
     * 微信号
     */
    private String wechat;

    /**
     * 出生日期
     */
    private Date birthday;

    /**
     * 头像
     */
    private String avatar;

    /**
     * 最后的登陆时间
     */
    private Date lasted;


}
