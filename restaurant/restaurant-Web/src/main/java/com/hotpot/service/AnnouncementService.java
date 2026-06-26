package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.Announcement;

import java.util.List;

public interface AnnouncementService extends IService<Announcement> {

    List<Announcement> listByStoreId(Long storeId);

    List<Announcement> listActive();
}
