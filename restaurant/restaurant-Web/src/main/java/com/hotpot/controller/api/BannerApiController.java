package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Banner;
import com.hotpot.service.BannerService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Api(tags = "C端-轮播图接口")
@RestController
@RequestMapping("/api/banners")
@RequiredArgsConstructor
public class BannerApiController {

    private final BannerService bannerService;

    @GetMapping("list")
    @ApiOperation("获取轮播图列表")
    public Result<List<Banner>> list() {
        return Result.success(bannerService.listVisible());
    }
}
