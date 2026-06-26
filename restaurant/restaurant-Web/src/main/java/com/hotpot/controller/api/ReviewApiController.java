package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Review;
import com.hotpot.service.ReviewService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import springfox.documentation.annotations.ApiIgnore;

@Api(tags = "C端-评价接口")
@RestController
@RequestMapping("/api/reviews")
@RequiredArgsConstructor
public class ReviewApiController {

    private final ReviewService reviewService;

    @ApiOperation("提交评价")
    @PostMapping
    public Result<?> create(@ApiParam("评价信息") @RequestBody Review review,
                            @ApiIgnore Authentication authentication) {
        Long customerId = Long.parseLong(authentication.getName());
        review.setCustomerId(customerId);
        reviewService.createReview(review);
        return Result.success();
    }
}
