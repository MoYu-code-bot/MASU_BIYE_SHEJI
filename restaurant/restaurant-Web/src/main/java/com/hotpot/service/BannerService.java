package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.Banner;

import java.util.List;

public interface BannerService extends IService<Banner> {

    List<Banner> listVisible();
}
