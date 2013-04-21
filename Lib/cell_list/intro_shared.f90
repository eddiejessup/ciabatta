module cell_list_shared

    use utils
    implicit none

    integer, allocatable :: inters(:, :), intersi(:)
    integer, parameter :: m_max = 100
    ! maximum number of possible interaction particles
    integer, parameter :: inters_max = 200

contains

subroutine initialise_inters(n)
    integer, intent(in) :: n

    if (allocated(inters)) deallocate(inters)
    if (allocated(intersi)) deallocate(intersi)
    if (.not. allocated(inters)) allocate(inters(min(n, inters_max), n))
    if (.not. allocated(intersi)) allocate(intersi(size(inters, 2)))
end subroutine

end module
