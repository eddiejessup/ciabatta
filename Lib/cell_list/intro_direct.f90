module cell_list_direct

    use utils
    use cell_list_shared
    implicit none

contains

subroutine make_inters(r, l, r_cut)
    real(dp), intent(in) :: r(:, :), l, r_cut
    integer :: i, i_target, m
    real(dp) :: l_half, r_cut_sq

    m = int(floor(l / r_cut))
    l_half = l / 2.0_dp
    r_cut_sq = r_cut ** 2
    call initialise_inters(size(r, 2))

    intersi = 0
    do i = 1, size(r, 2)
        do i_target = i + 1, size(r, 2)
            if (r_sep_sq(r(:, i), r(:, i_target), l, l_half) < r_cut_sq) then
                intersi(i) = intersi(i) + 1
                inters(intersi(i), i) = i_target
                intersi(i_target) = intersi(i_target) + 1
                inters(intersi(i_target), i_target) = i
            end if
        end do
    end do

end subroutine

end module
