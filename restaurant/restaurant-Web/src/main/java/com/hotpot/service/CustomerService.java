package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.Customer;

public interface CustomerService extends IService<Customer> {

    String login(String phone, String password);

    String register(String phone, String password, String nickname);

    Customer getByPhone(String phone);
}
