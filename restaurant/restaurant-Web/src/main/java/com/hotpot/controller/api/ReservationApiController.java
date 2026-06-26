package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Reservation;
import com.hotpot.service.ReservationService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import springfox.documentation.annotations.ApiIgnore;

import java.util.List;

@Api(tags = "C端-预订接口")
@RestController
@RequestMapping("/api/reservations")
@RequiredArgsConstructor
public class ReservationApiController {

    private final ReservationService reservationService;

    @ApiOperation("创建预订")
    @PostMapping
    public Result<String> create(@ApiParam("预订信息") @RequestBody Reservation reservation,
                                  @ApiIgnore Authentication authentication) {
        Long customerId = Long.parseLong(authentication.getName());
        reservation.setCustomerId(customerId);
        String orderNo = reservationService.createReservation(reservation);
        return Result.success(orderNo);
    }

    @ApiOperation("我的预订列表")
    @GetMapping
    public Result<List<Reservation>> list(@ApiIgnore Authentication authentication) {
        Long customerId = Long.parseLong(authentication.getName());
        return Result.success(reservationService.listByCustomerId(customerId));
    }

    @ApiOperation("预订详情")
    @GetMapping("/detail")
    public Result<Reservation> detail(@ApiParam("预订ID") @RequestParam Long reservationId) {
        return Result.success(reservationService.getById(reservationId));
    }

    @ApiOperation("取消预订")
    @PutMapping("/cancel")
    public Result<?> cancel(@ApiParam("预订ID") @RequestParam Long reservationId,
                            @ApiParam("取消原因") @RequestParam(required = false) String reason,
                            @ApiIgnore Authentication authentication) {
        Long customerId = Long.parseLong(authentication.getName());
        reservationService.cancelReservation(reservationId, customerId, reason);
        return Result.success();
    }
}
